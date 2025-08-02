import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from tqdm import tqdm
from reuse import *
from datetime import datetime
import numpy as np
import multiprocessing

def main():
    st.set_page_config(page_title="Conversation Insights", page_icon="📖", layout="wide")

    st.title(":clipboard: :blue[Conversation Insights]") 
    st.caption(" :blue[Intelligent Conversation Retrieval System(iCRS)]")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a transcript(.pdf or .docx) or an audio file",
        type=["wav", "m4a", "mp3"],
        help="Scanned file is not supported yet!",
    ) 

    col1, col2 = st.columns([1,1])
    if uploaded_file is not None:
        path_in = uploaded_file.name
        file_path = "data/"
        audio_file = open(file_path+path_in, 'rb')
        audio_stream = audio_file.read()
        with col1:
            st.audio(audio_stream)
        with col2:
            translate_button = st.button("Translate", type="primary")
        if translate_button:
            st.header("Conversation Transcript")
            conversation = convert_speech_to_text(path_in)
            conv_pdf = create_pdf(conversation, "artifacts/Conversation_Transcript.pdf")
            with st.expander("**:blue[Transcript]**", expanded=False):
                st.write(conversation)
    else:
        path_in = None

    st.divider()
    
    pdf_file = "data/Kannada_Gen_Transcript.pdf"

    retrieve_button = st.button("Retrieve", type="primary")
    st.header("Conversation Overview")
    if retrieve_button:

        conversation_queue = multiprocessing.Queue()
        customer_info_queue = multiprocessing.Queue()
        aspects_and_sentiments_queue = multiprocessing.Queue()
        competitor_comparison_queue = multiprocessing.Queue()

        processes = [
        #Extract Entities and Relationships
        multiprocessing.Process(target=extract_entities_and_relationships, args=(customer_info_queue,conversation_queue)),
        
        #Extract Sentiments
        multiprocessing.Process(target=extract_sentiments, args=(aspects_and_sentiments_queue,)),
        
        #Competitor Comparison
        multiprocessing.Process(target=extract_competitor_comparison, args=(competitor_comparison_queue,))
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        convsummary = conversation_queue.get()
        customer_info = customer_info_queue.get()
        aspects_and_sentiments = aspects_and_sentiments_queue.get()
        competitor_comparison = competitor_comparison_queue.get()

        customer_info_json = json.loads(customer_info)

        data = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Customer Name': customer_info_json.get("customer_name"),
            'Customer Email ID': customer_info_json.get("customer_email_id"),
            'Customer Mobile No': customer_info_json.get("customer_mobile_no"),
            'Relationship Manager': customer_info_json.get("rm_name"),
            'Model Preference': customer_info_json.get("customer_preference"),
            'Variant Preference': customer_info_json.get("variant_preference"),
            'Color Preference': customer_info_json.get("color_preference"),
            'Customer Sentiment': aspects_and_sentiments, 
            'Conversation Summary' : convsummary,
            'Competitor Comparison' : competitor_comparison
        }

        # Create a DataFrame and update record
        df_new = pd.DataFrame(data,index=[0])
        df_existing = pd.read_csv("data/extracted_data.csv")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv("data/extracted_data.csv", index=False)

        gb = GridOptionsBuilder.from_dataframe(df_combined)
        gb.configure_selection(selection_mode="single", use_checkbox=True)
        gb.configure_pagination(enabled=True)
        gb.configure_default_column(
            resizable=True,
            filterable=True,
            sortable=True,
            editable=False,
        )
        gridoptions = gb.build()
        grid_table = AgGrid(
                df_combined,
                gridOptions=gridoptions,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=300,
                #theme='fresh'
            )
    else:
        df_existing = pd.read_csv("data/extracted_data.csv")
        gb = GridOptionsBuilder.from_dataframe(df_existing)
        gb.configure_selection(selection_mode="single", use_checkbox=True)
        gb.configure_pagination(enabled=True)
        gb.configure_default_column(
            resizable=True,
            filterable=True,
            sortable=True,
            editable=False,
        )
        gridoptions = gb.build()
        grid_table = AgGrid(
                df_existing,
                gridOptions=gridoptions,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=300,
                #theme='fresh'
            )

    # Assuming grid_table is your DataFrame containing the selected rows
    selected_rows = grid_table["selected_rows"]

    if not selected_rows.empty:
        st.header("Conversation Details")
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("**:blue[Customer Information]**", expanded=False):
                st.write(":label: **:grey[Name]**: ", selected_rows.iloc[0]["Customer Name"])
                st.write(":envelope_with_arrow: **:grey[Email]**: ", selected_rows.iloc[0]["Customer Email ID"])
                st.write(":telephone_receiver: **:grey[Contact]**: ", selected_rows.iloc[0]["Customer Mobile No"])

        with col2:
            with st.expander("**:blue[Vehicle Information]**", expanded=False):
                st.write("**:grey[Model]**: ", selected_rows.iloc[0]["Model Preference"])
                st.write("**:grey[Variant]**: ", selected_rows.iloc[0]["Variant Preference"])
                st.write("**:grey[Color]**: ", selected_rows.iloc[0]["Color Preference"])

        col3, col4 = st.columns(2)
        with col3:
            with st.expander("**:blue[Conversation Summary]**", expanded=False):
                header = f"<b>Conversed with Relationship Manager</b> {selected_rows.iloc[0]['Relationship Manager']} <b>on</b> {selected_rows.iloc[0]['Timestamp']}"
                details = selected_rows.iloc[0]["Conversation Summary"]
                # Create the Markdown string with HTML tags
                markdown_text = f"<p>{header}</p>\n<p>{details}</p>"

                # Display the Markdown string
                st.markdown(markdown_text, unsafe_allow_html=True)      

        with col4:
            with st.expander("**:blue[Customer Sentiments]**", expanded=False):
                st.write(selected_rows.iloc[0]["Customer Sentiment"])
                #st.write(format_sentiments(sentiment_list))
        
        with st.expander("**:blue[Competitor Comparison]**", expanded=False):
            st.write(selected_rows.iloc[0]["Competitor Comparison"] )
    else:
        st.write("No rows selected.")

    update_button = st.button("Update CDP", type="primary")
    if not uploaded_file:
        st.stop()

if __name__=="__main__":
    main()

