import os
import re
import json
import openai
import PyPDF2
from fpdf import FPDF
from tqdm import tqdm
import streamlit as st
from langchain.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
    )
from langchain.graphs import Neo4jGraph
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from langchain.pydantic_v1 import Field, BaseModel
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import GraphCypherQAChain
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from openai import OpenAI
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


def extract_entities_and_relationships(customer_info_queue, conversation_queue):
    convsummary = ""
    customer_info = {}
    conv_pdf = "artifacts/Conversation_Transcript.pdf"
    convsummary = summarize_conversation(conv_pdf)

    conversation_queue.put(convsummary)
    summary_pdf = create_pdf(convsummary, "artifacts/Summary_Transcript.pdf")

    customer_info  = extract_key_entities(summary_pdf)

    customer_info_queue.put(customer_info)
    
def extract_sentiments(aspects_and_sentiments_queue):
    aspects_and_sentiments = ""
    conv_pdf = "artifacts/Conversation_Transcript.pdf"
    conversation = read_pdf(conv_pdf)
    aspects_and_sentiments = extract_sentiments_from_conversation(conversation)

    aspects_and_sentiments_queue.put(aspects_and_sentiments)
    
def extract_competitor_comparison(competitor_comparison_queue):
    competitor_comparison_text = ""
    competitor_comparison_text = compare_competitors("MG Comet EV")

    competitor_comparison_queue.put(competitor_comparison_text)

def convert_speech_to_text(audiofile):
    openai.api_key = os.environ['OPENAI_API_KEY']
    with open("data/"+audiofile, "rb") as audio_file:
        client = OpenAI()
        translated_text = client.audio.transcriptions.create(
            file = audio_file,
            model = "whisper-1",
            response_format="text",
            language="en"
            )
    return translated_text

def create_pdf(text, file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add each line of text to the PDF
    lines = text.split('\n')
    for line in lines:
        pdf.cell(50, 5, txt=line, ln=True)

    # Save the PDF to the specified file path
    pdf.output(file_path)

def read_pdf(file):
    with open(file, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_key_entities(summary_pdf):

    #Neo4j Instance
    url = os.environ["NEO4J_URL"]
    username ="neo4j"
    password = os.environ['NEO4J_PWD']
    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password
    )

    #LLM Instance
    proxy_client = get_proxy_client('gen-ai-hub')
    llm = ChatOpenAI(proxy_model_name='gpt-4-32k', proxy_client=proxy_client)

    class Property(BaseModel):
        key: str = Field(..., description="key")
        value: str = Field(..., description="value")

    class Node(BaseNode):
        properties: Optional[List[Property]] = Field(
            None, description="List of node properties")

    class Relationship(BaseRelationship):
        properties: Optional[List[Property]] = Field(
            None, description="List of relationship properties"
        )

    class KnowledgeGraph(BaseModel):
        nodes: List[Node] = Field(
            ..., description="List of nodes in the knowledge graph")
        rels: List[Relationship] = Field(
            ..., description="List of relationships in the knowledge graph"
        )

    def format_property_key(s: str) -> str:
        words = s.split()
        if not words:
            return s
        first_word = words[0].lower()
        cap_words = [word.capitalize() for word in words[1:]]
        return "".join([first_word] + cap_words)

    def convert_properties_to_dict(props) -> dict:
        properties = {}
        if not props:
            return properties
        for p in props:
            properties[format_property_key(p.key)] = p.value
        return properties

    def map_kg_node_to_base_node(node: Node) -> BaseNode:
        properties = convert_properties_to_dict(node.properties) if node.properties else {}
        properties["name"] = node.id.title()
        return BaseNode(
            id=node.id.title(), type=node.type.capitalize(), properties=properties
        )

    def map_kg_rel_to_base_relationship(rel: Relationship) -> BaseRelationship:
        source = map_kg_node_to_base_node(rel.source)
        target = map_kg_node_to_base_node(rel.target)
        properties = convert_properties_to_dict(rel.properties) if rel.properties else {}
        return BaseRelationship(
            source=source, target=target, type=rel.type, properties=properties
        )

    def get_extraction_chain(
        allowed_nodes: Optional[List[str]] = None,
        allowed_rels: Optional[List[str]] = None
        ):
        prompt = ChatPromptTemplate.from_messages(
            [(
            "system",
            f"""# Knowledge Graph Instructions
            ## 1. Overview
            You are an expert in building a Knowledge Graph. 
            Your goal is to convert unstructured documents to a Knowledge Graph with the key elements like entities, relationships and their properties. 
            The resulting Knowledge Graph (KG) should capture all the esense of customer conversation with relationship manager.
            - **Nodes** represent entities and concepts.

            ## 2. Labeling Nodes
            - **Consistency**: Ensure you use basic or elementary types for node labels.
            - **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
            {'- **Allowed Node Labels:**' + ", ".join(allowed_nodes) if allowed_nodes else ""}
            {'- **Allowed Relationship Types**:' + ", ".join(allowed_rels) if allowed_rels else ""}

            ## 3. Handling Numerical Data and Dates
            - Numerical data, like age, phone number, should be incorporated as attributes or properties of the respective nodes.
            - **Property Format**: Properties must be in a key-value format.

            ## 4. Coreference Resolution
            - **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
            If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
            always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
            Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.

            ## 5. Strict Compliance
            Adhere to the rules strictly. Non-compliance will result in termination.
                    """),
                        ("human", "Use the given format to extract information from the following input: {input}"),
                        ("human", "Tip: Make sure to answer in the correct format"),
                    ])

        return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)

    def extract_and_store_graph(
        document: Document,
        nodes:Optional[List[str]] = None,
        rels:Optional[List[str]]=None) -> None:
        # Extract graph data using OpenAI functions
        extract_chain = get_extraction_chain(nodes, rels)
        data = extract_chain.run(document.page_content)
        # Construct a graph document
        graph_document = GraphDocument(
        nodes = [map_kg_node_to_base_node(node) for node in data.nodes],
        relationships = [map_kg_rel_to_base_relationship(rel) for rel in data.rels],
        source = document
        )
        # Store information into a graph
        graph.add_graph_documents([graph_document])
    
    # Document Loader - Imports and reads data
    loader = PyPDFLoader("artifacts/Summary_Transcript.pdf")
    documents = loader.load()

    # Define chunking strategy
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=24)

    # Only take the first the raw_documents
    documents = text_splitter.split_documents(documents)

    # Delete the graph
    graph.query("MATCH (n) DETACH DELETE n")

    # Specify which node and relationship labels should be extracted by the LLM
    allowed_nodes = ["Customer","Relationship_Manager","Car","Car_Delearship","Inventory","Test_Drive","Price_Negotiation","Documentation","Delivery","Premium_Features","Customer_Service","Feedback_Reviews","Marketing_Promotions","Competition","Extended_Warranty","Accessories","Insurance", "Performance_Features","Driver_Assistance_Features","Storage_Cargo_Space", "Safety_Features","Safety_Ratings","Fuel_Efficiency","Comfort_Features","Infotainment_System","Advanced_Technology_Features","Interiors","Exteriors", "Model", "Variant","Finance_Options","Alternate_Options","Booking_Details", "Expected_Delivery_Date"]
    allowed_rels = ["INTERACTS_WITH", "INQUIRES_ABOUT", "SELECTS_MODEL", "SELECTS_VARIANT", "MAKES_PURCHASE", "TEST_DRIVES", "OWNS", "SERVICE_AT", "HAS_FINANCIAL_AGREEMENT", "PREFERS", "RECOMMENDS", "REFERS_TO", "VISITED", "MET_WITH", "CONTACTED", "VISITS_SHOWROOM", "NEGOTIATES_PRICE", "ATTENDS_EVENT", "UPGRADES_VEHICLE", "USES_FINANCING", "PROVIDES_FEEDBACK", "REQUESTS_SERVICE", "PARTICIPATES_IN_LOYALTY_PROGRAM", "JOINS_COMMUNITY", "ATTENDS_TRAINING", "SUBMITS_REVIEW","CONSIDERS","CONSIDERS_ALTERNATE_OPTIONS","SERVICE_AT","UPGRADES_VEHICLE"]
    for i, d in tqdm(enumerate(documents), total=len(documents)):
        extract_and_store_graph(d, allowed_nodes,allowed_rels)

    graph.refresh_schema()
    
    cypher_chain = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm = llm,
        qa_llm = llm,
        validate_cypher=True, # Validate relationship directions
        verbose=False,
        return_direct=True
    )

    try:
        customer_name = next(iter(cypher_chain.run("What is the name of the customer?")[0].values()), None),
    except Exception as e:
        print("Error:", e)
        customer_name = None 

    try:
        rm_name = next(iter(cypher_chain.run("What is the name of Salesperson who attended customer?")[0].values()), None)
    except Exception as e:
        print("Error:", e)
        rm_name = None

    try:
        customer_mobile_no = next(iter(cypher_chain.run("What is the mobile no for customer?")[0].values()), None)
    except Exception as e:
        print("Error:", e)
        customer_mobile_no = None
    
    try:
        customer_email_id = next(iter(cypher_chain.run("What is the email id for customer?")[0].values()), None)
    except Exception as e:
        print("Error:", e)
        customer_email_id = None

    try:
        customer_preference = next(iter(cypher_chain.run("What is the model name?")[0].values()), None)
    except Exception as e:
        print("Error:", e)
        customer_preference = None

    try:
        variant_preference = next(iter(cypher_chain.run("What is the model variant?")[0].values()), None)
    except Exception as e:
        print("Error:", e)
        variant_preference = None

    try:
        color_preference = next(iter(cypher_chain.run("What is the model color?")[0].values()), None)
    except Exception as e:
        print("Error:", e)
        color_preference = None
        
    customer_info = {
        "customer_name": customer_name,
        "rm_name": rm_name,
        "customer_mobile_no": customer_mobile_no,
        "customer_email_id": customer_email_id,
        "customer_preference": customer_preference,
        "variant_preference": variant_preference,
        "color_preference": color_preference
        }
    customer_info_json = json.dumps(customer_info)
    return customer_info_json

def summarize_conversation(pdf_file):
    
    #LLM Instance
    proxy_client = get_proxy_client('gen-ai-hub')
    llm = ChatOpenAI(proxy_model_name='gpt-4-32k', proxy_client=proxy_client)

    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=24)
    documents = text_splitter.split_documents(documents)
    
    text = read_pdf(pdf_file)
    
    summary_prompt_template = """
    You are an expert at Summarizing Conversations and you will be given a series of a conversations in a dealership. 
    Your goal is to give a precise verbose summary in a paragraph containing the following elements MANDATORILY. 
    1. Customers Name, 
    2. Relationship Manager Name/Sales Representative Name,
    3. Customer Mobile Number details, 
    4. Customers Email ID details, 
    5. Model which the customer is looking for, 
    6. Variant which the customer is looking for, 
    7. Color which the customer is looking for,
    8. Customers Feedback highlighting Positives, Negatives and Neutral aspects.
    9. Customers Alternate Options.
    10. All enquiries about the car features, car interiors and exteriors, car performance and car finance by the Customer.
    11. Details about the Test Drive Request from Customer.

    ```{text}```
    VERBOSE SUMMARY:
    """
    final_summary_prompt_template = PromptTemplate(template=summary_prompt_template, input_variables=["text"])
    summary_chain = load_summarize_chain(llm=llm, prompt=final_summary_prompt_template, chain_type='stuff',verbose=False)

    convsummary = summary_chain.run(documents)
    return convsummary

@st.cache_data()
def extract_sentiments_from_conversation(conversation):
    
    proxy_client = get_proxy_client('gen-ai-hub')
    llm = ChatOpenAI(proxy_model_name='gpt-4-32k', proxy_client=proxy_client)
    
    #Retreive aspects from Conversation
    examples_aspect = [{
    "conversation":'''MG car impresses with its cutting-edge technology, sleek design, and electric powertrain, providing a dynamic driving experience. Its luxurious interior with the high-quality materials, gives the car a premium feel and comfort. Its advanced safety features and competitive pricing make it a compelling choice in the automotive market. Great value for the money''',
    "aspects": '''Technology and Features, Interior and Exterior Design, Comfort, Safety, Value for Money, Pricing'''
    },
    {
    "conversation":'''MG car falls short in terms of build quality and reliability, with frequent mechanical issues reported by users. The fuel efficiency doesn't come close to what was advertised. The subpar performance, coupled with inconsistent customer service, makes it a less appealing option in the competitive automotive landscape. Dealing with maintenance has been a nightmare – expensive and time-consuming.The resale value seems to be dropping fast. I wish I had considered other options before buying.''',
    "aspects": '''Build Quality, Reliability, Fuel Efficiency, Performance, Customer Service, Maintenance, Resale Value'''
    }
    ]
    
    prompt_template_aspect = '''
    Review: {conversation}
    {aspects}
    '''
    example_prompt_aspect = PromptTemplate(input_variables=['conversation','aspects'], template=prompt_template_aspect)
    final_prompt_aspect = FewShotPromptTemplate(
                    examples=examples_aspect,
                    example_prompt=example_prompt_aspect,
                    suffix="Conversation: {conversation}/n",
                    input_variables=["conversation"],
                    prefix="You have to extract aspects from the Customer Conversation with Relationship Manager. Take conversation as input and extract various aspects like Technology and Features, Interior and Exterior Design, Comfort, Safety, Value for Money, Pricing, Build Quality, Reliability, Fuel Efficiency, Performance, Customer Service, Maintenance, Resale Value. Finally return these aspects as a list")

    aspect_extraction_chain = LLMChain(llm=llm, prompt=final_prompt_aspect, output_key="aspects")
    aspects = aspect_extraction_chain.predict(conversation="Good morning, sir. Welcome to MG Motor. I'm Praveen the relationship manager. How can I help you today?Hello, sir. Namaste. My name is Akshath and I am looking for MG Comet EV vehicle. Can you tell me about it? Definitely, sir. I'll be happy to help you. Before I help you, can I get your mobile number and DIN number? Yes, sir. My mobile number is 9740680519. You can email me at akshat.ltrgv.com. Thank you, sir. Which specific model are you looking at in Comet? Sir, I'm looking at MG Comet EV Plus variant. Can you tell me about it? Can you confirm the specific color? Sir, I'm looking at Apple Green color. Thank you, sir. That helps. MG Comet EV is one of the latest electric vehicles. It has sleek design and advanced technology features. The electric powertrain is one of the powerful ones in the industry. Can you tell me more about it? Yes, sir. I'm looking at the performance, acceleration and range. MG Comet EV is one of the most powerful and impressive vehicles in the industry. It has an electric motor and advanced technology features. It can reach speeds from 0 to 60 miles per hour in 7 seconds. It can reach 250 miles per hour in full charge. It can be a reliable partner for daily use and long trips. Thank you, sir. It sounds promising. Do you have any charging options? Do you have fast charging options? Do you have a battery charge option? Sure, sir. There is a fast charging option. MG Comet EV has DC fast charging. You can get 80% charge in 30 minutes for quick recharging. If you use standard AC charging for home charging, it will take more time. It depends on the output power. On average, you can get a full charge overnight. What are the features of MG Comet EV? What are the features of MG Comet EV? What are the safety features? You will get state-of-the-art infotainment system. You will get latest infotainment system. You will get smart phone integration and navigation. You can operate it with voice commands. You will get advanced driver assistance features. You will get departure warning. You will get automatic emergency braking. You will get adaptive cruise control. Please tell us about warranty and maintenance. You will get comprehensive warranty package from MG Comet EV. You will get comprehensive warranty package from MG Comet EV. You will get comprehensive warranty package from MG Comet EV. You can go to the Finalized section. You will get comprehensive warranty package from MG Comet EV. Please share with us the color options of MG Comet EV. There are many colors available. Basic, classic shades like midnight black, arctic white, electric blue. I will show you samples and give you options to select. I appreciate it. Do you have any promotions or discounts? Please tell us about the options in financing. Currently, we are doing special financing promotion for MG Comet EV. You will get attractive interest rates. We provide flexible plans in payment options. I will give you more details about the best financing options. I will share with you if you need it or not. Thank you sir. Before I go to the final decision, can I see the test drive? You can arrange the test drive as per your convenience. We will make sure that you get the Comet EV for the on-road experience. When is the best time for the test drive? Can you arrange it next Monday morning? I will make a schedule for the test drive on Sunday morning. I will be more than happy to help you. I will be around to assist you. I will book the test drive for you. Thank you sir. I am looking forward to the test drive. I will give you feedback after the test drive. Thank you sir. If you have any questions, do not hesitate to contact me. I will be more than happy to help you. Thank you sir.")
    
     #Retreive sentiments from Conversation
    examples_sentiment = [{
                "conversation":'''My MG has been incredibly reliable. I've driven it for years without any major issues.''',
                "aspects": '''Reliability''',
                "aspects_with_sentiment": '''(Reliability, Positive)'''
                },
                {
                "conversation":'''I've had several breakdowns within the first year of owning my MG. Reliability is a major concern.''',
                "aspects": '''Reliability''',
                "aspects_with_sentiment": '''(Reliability, Negative)'''
                },
                {
                "conversation":'''The performance of my MG is outstanding. It accelerates smoothly, handles well, and makes every drive enjoyable''',
                "aspects": '''Performance''',
                "aspects_with_sentiment": '''(Performance, Positive)'''
                },
                {
                "conversation":'''The performance is disappointing. It feels sluggish, and the handling is subpar.The performance is lackluster''',
                "aspects": '''Performance''',
                "aspects_with_sentiment": '''(Performance, Negative)'''
                }
                ]
    
    prompt_template_sentiment='''
    Given the conversation and extracted aspects, determine the sentiments for these aspects e.g. 'Positive' or 'Negative' or 'Neutral' in the following format (aspect, sentiment).
    {conversation}
    {aspects}
    [["aspects", "aspects_with_sentiment"]]
    '''
    sentiment_list = []
    example_prompt_sentiment=PromptTemplate(input_variables=["conversation","aspects"], template=prompt_template_sentiment)

    final_prompt_sentiment = FewShotPromptTemplate(
                    examples=examples_sentiment,
                    example_prompt=example_prompt_sentiment,
                    suffix="Conversation: {conversation}/n",
                    input_variables=["conversation"],
                    prefix="We have to determine the sentiment for each of the extracted aspects from the Customer Conversation with Relationship Manager. Take conversation and aspects as input and determine sentiments for each of the aspects like Technology and Features, Interior and Exterior Design, Comfort, Safety, Value for Money, Pricing, Build Quality, Reliability, Fuel Efficiency, Performance, Customer Service, Maintenance, Resale Value. Finally return the sentiments for each of the aspect as a list"
                    )
    
    aspect_sentiment_chain=LLMChain(llm=llm,prompt=final_prompt_sentiment,output_key="aspects_with_sentiment")
    sentiment = aspect_sentiment_chain.predict(conversation="Good morning, sir. Welcome to MG Motor. I'm Praveen the relationship manager. How can I help you today?Hello, sir. Namaste. My name is Akshath and I am looking for MG Comet EV vehicle. Can you tell me about it? Definitely, sir. I'll be happy to help you. Before I help you, can I get your mobile number and DIN number? Yes, sir. My mobile number is 9740680519. You can email me at akshat.ltrgv.com. Thank you, sir. Which specific model are you looking at in Comet? Sir, I'm looking at MG Comet EV Plus variant. Can you tell me about it? Can you confirm the specific color? Sir, I'm looking at Apple Green color. Thank you, sir. That helps. MG Comet EV is one of the latest electric vehicles. It has sleek design and advanced technology features. The electric powertrain is one of the powerful ones in the industry. Can you tell me more about it? Yes, sir. I'm looking at the performance, acceleration and range. MG Comet EV is one of the most powerful and impressive vehicles in the industry. It has an electric motor and advanced technology features. It can reach speeds from 0 to 60 miles per hour in 7 seconds. It can reach 250 miles per hour in full charge. It can be a reliable partner for daily use and long trips. Thank you, sir. It sounds promising. Do you have any charging options? Do you have fast charging options? Do you have a battery charge option? Sure, sir. There is a fast charging option. MG Comet EV has DC fast charging. You can get 80% charge in 30 minutes for quick recharging. If you use standard AC charging for home charging, it will take more time. It depends on the output power. On average, you can get a full charge overnight. What are the features of MG Comet EV? What are the features of MG Comet EV? What are the safety features? You will get state-of-the-art infotainment system. You will get latest infotainment system. You will get smart phone integration and navigation. You can operate it with voice commands. You will get advanced driver assistance features. You will get departure warning. You will get automatic emergency braking. You will get adaptive cruise control. Please tell us about warranty and maintenance. You will get comprehensive warranty package from MG Comet EV. You will get comprehensive warranty package from MG Comet EV. You will get comprehensive warranty package from MG Comet EV. You can go to the Finalized section. You will get comprehensive warranty package from MG Comet EV. Please share with us the color options of MG Comet EV. There are many colors available. Basic, classic shades like midnight black, arctic white, electric blue. I will show you samples and give you options to select. I appreciate it. Do you have any promotions or discounts? Please tell us about the options in financing. Currently, we are doing special financing promotion for MG Comet EV. You will get attractive interest rates. We provide flexible plans in payment options. I will give you more details about the best financing options. I will share with you if you need it or not. Thank you sir. Before I go to the final decision, can I see the test drive? You can arrange the test drive as per your convenience. We will make sure that you get the Comet EV for the on-road experience. When is the best time for the test drive? Can you arrange it next Monday morning? I will make a schedule for the test drive on Sunday morning. I will be more than happy to help you. I will be around to assist you. I will book the test drive for you. Thank you sir. I am looking forward to the test drive. I will give you feedback after the test drive. Thank you sir. If you have any questions, do not hesitate to contact me. I will be more than happy to help you. Thank you sir.")
    
    overall_chain = SequentialChain(
                    chains = [aspect_extraction_chain,aspect_sentiment_chain],
                    input_variables = ["conversation"],
                    output_variables = ["aspects_with_sentiment"],
                    verbose=True
                    )
    output_data = overall_chain(conversation)
    
    # Check if the key 'aspects_with_sentiment' exists in the dictionary
    if 'aspects_with_sentiment' in output_data:
        print("aspects_with_sentiment is present")
        try:
            
            sentiment_list = json.loads(re.search(r'\[\[.*?\]\]|\[.*?\]', output_data['aspects_with_sentiment']).group())
            #sentiment_list = json.loads(re.search(r'\[.*\]', output_data['aspects_with_sentiment']).group())
            aspects_and_sentiments = format_sentiments(sentiment_list)
            return aspects_and_sentiments
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            return ""
    else:
        print("aspects_with_sentiment is not present")
        return ""

def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )
    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

@st.cache_data()
def format_sentiments(sentiment_list):
    result_dict = {"Positive": [], "Neutral": [], "Negative": []}

    for aspect_sentiment_pair in sentiment_list:
        aspect = aspect_sentiment_pair[0]
        sentiment = aspect_sentiment_pair[1]
        result_dict[sentiment].append(aspect)

    output = "**:grey[Positive:]** :blush: \n"
    output += "\n".join([f"{i}. {aspect}" for i, aspect in enumerate(result_dict["Positive"], start=1)])

    output += "\n\n**:grey[Neutral:]** :neutral_face:\n"
    output += "\n".join([f"{i}. {aspect}" for i, aspect in enumerate(result_dict["Neutral"], start=1)])

    output += "\n\n**:grey[Negative:]** :worried: \n"
    output += "\n".join([f"{i}. {aspect}" for i, aspect in enumerate(result_dict["Negative"], start=1)])
    return output

def compare_competitors(model):
    output_parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(
            "You are an expert in automotive marketing. Your task is to conduct a comprehensive comparison between {car} and its competitor, emphasizing the superior qualities of MG vehicles over their rivals. The objective of this comparison is to persuasively demonstrate to potential customers why choosing an MG car is the preferable option over other available vehicles")
    proxy_client = get_proxy_client('gen-ai-hub')
    llm = ChatOpenAI(proxy_model_name='gpt-4-32k', proxy_client=proxy_client)
    chain = LLMChain(
            prompt = prompt,
            llm=llm,
            output_parser = output_parser)
    comparison_text = chain.run(model)
    return comparison_text
