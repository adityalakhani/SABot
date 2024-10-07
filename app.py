import streamlit as st
import logging
import os
import tempfile
import shutil

# import
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.llama_api import LlamaAPI
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import StorageContext
from IPython.display import Markdown, display
import qdrant_client
from typing import Optional
from PyPDF2 import PdfReader
import os
from PIL import Image

def extract_text_and_images(data_dir, img_dir, text_dir):
    # List all PDF files in the data directory
    pdfs = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]

    # Ensure that image and text directories exist
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    for pdf in pdfs:
        pdf_path = os.path.join(data_dir, pdf)
        text, images = _extract_from_pdf(pdf_path)

        # Save the extracted text and images
        _save_text(pdf, text, text_dir)
        _save_images(pdf, images, img_dir)

def _extract_from_pdf(pdf_path):
    text = ""
    images = []

    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]

            # Extract text
            page_text = page.extract_text()
            if page_text:  # To avoid adding None values if no text is extracted
                text += page_text

            # Extract images
            if hasattr(page, 'images'):
                for image in page.images:
                    images.append(image)

    return text, images

def _save_text(pdf, text, text_dir):
    # Remove .pdf extension from the filename
    base_filename = os.path.basename(pdf).replace('.pdf', '')
    text_file_path = os.path.join(text_dir, f'{base_filename}.txt')

    # Save the extracted text
    with open(text_file_path, 'w', encoding="utf-8") as f:
        f.write(f"{base_filename} starts here\n")
        f.write(text)

def _save_images(pdf, images, img_dir):
    base_filename = os.path.basename(pdf).replace('.pdf', '')

    for i, image in enumerate(images):
        # Generate a unique filename for each image
        image_file_name = f"{base_filename}_{i}"
        image_extension = image.name.split(".")[-1] if '.' in image.name else 'jpg'  # Use default jpg if extension is missing
        image_path = os.path.join(img_dir, f'{image_file_name}.{image_extension}')

        # Save image data
        with open(image_path, 'wb') as f:
            f.write(image.data)

def image_check(path):
    files = os.listdir(path)
    for file in files:
        image = Image.open(path+file)

        width, height = image.size
        if width < 101 and height < 101:
            image.close()
            os.remove(path+file)

def create_db(text_store, image_store, text_dir, image_dir) -> tuple:
    api_key = "LL-zrPJtSllWuKsJpC13YSelp66yjL8Oz5xZju7LCqoPGe0QVriEffjIQGGcowC9FIs"
    llm = LlamaAPI(api_key=api_key, temperature=0.2, max_tokens=512)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    text_storage_context = StorageContext.from_defaults(
        vector_store=text_store
    )

    text_documents = SimpleDirectoryReader(text_dir).load_data()
    text_index = VectorStoreIndex.from_documents(
        documents=text_documents,
        storage_context=text_storage_context, 
        embed_model=embed_model, 
        llm = llm
    )


    image_storage_context = StorageContext.from_defaults(
        image_store=image_store
    )

    # Create the MultiModal index
    image_documents = SimpleDirectoryReader(image_dir).load_data()
    image_index = MultiModalVectorStoreIndex.from_documents(
        image_documents,
        storage_context=image_storage_context,
        embed_model=embed_model
    )
    return text_index, image_index


# Streamlit page configuration
st.set_page_config(
    page_title="SABot Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=True)
def process_uploads(file_upload):
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Uploaded the files: {file_upload}")
    logging.info(f'temporary storage location: {temp_dir}')
    for uploaded_file in file_upload:
        logger.info("Uploading a file")
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
            logger.info(f"File saved to temporary path: {path}")
    
    client = qdrant_client.QdrantClient(path=temp_dir+"/qdrant")
    extract_text_and_images(temp_dir, temp_dir+"/images", temp_dir+"/texts")
    logger.info(f"Extracted images and texts saved to '{temp_dir}/images/' and '{temp_dir}/texts'")

    image_check(temp_dir+"/images/")
    logger.info(f"Garbage images deleted")

    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection_0"
    )

    image_store = QdrantVectorStore(
        client=client, collection_name="image_collection_0"
    )
    create_db(text_store, image_store, temp_dir+"/texts/", temp_dir+"/images/")
    logger.info("Created 'text_collection_0' & image_collection_0' in qdrant")

    # shutil.rmtree(temp_dir)
    # logger.info(f"Temporary directory {temp_dir} removed")

    return client



@st.cache_data()
def delete_uploads(_client) -> None:
    logger.info("Deleting vector DB")
    try:
        collections = _client.get_collections().collections

        for collection in collections:
            _client.delete_collection(collection_name=collection.name)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()

    except Exception as e:
        st.write("Couldn't delete the collections.")


def main() -> None:

    st.subheader("SABot", divider="gray", anchor=False)
    
    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    
    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", type="pdf", accept_multiple_files=True
    )

    if file_upload:
        st.session_state["file_upload"] = file_upload
        if st.session_state["vector_db"] is None:
            st.session_state["vector_db"] = process_uploads(file_upload)
    
    delete_collection = col1.button("⚠️ Delete collection", type="secondary")

    if delete_collection:
        delete_uploads(st.session_state["vector_db"])

    with col2:
        message_container = st.container(height=500, border=True)

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar="😎").markdown(prompt)

                # with message_container.chat_message("assistant", avatar="🤖"):
                #     with st.spinner(":green[processing...]"):
                #         if st.session_state["vector_db"] is not None:
                            # response = process_question(
                            #     prompt, st.session_state["vector_db"]
                            # )
                            # st.markdown(response)
                        # else:
                        #     st.warning("Please upload a PDF file first.")

                # if st.session_state["vector_db"] is not None:
                #     st.session_state["messages"].append(
                #         {"role": "assistant", "content": response}
                #     )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to begin chat...") 


if __name__ == "__main__":
    main()
