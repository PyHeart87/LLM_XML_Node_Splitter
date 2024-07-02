import streamlit as st
from lxml import etree
import requests
import json
import chromadb
from chromadb.config import Settings
import re

# Initialize Chroma
chroma_client = chromadb.Client(Settings(persist_directory="./xml_splitter_db"))
collection = chroma_client.get_or_create_collection("xml_splits")

# Function to split XML node using CodeLlama
def split_xml_node(xml_content, node_path):
    # Remove XML declaration if present
    xml_content = re.sub(r'<\?xml[^>]+\?>', '', xml_content).strip()
    
    # Parse the XML content
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(xml_content.encode('utf-8'), parser=parser)
    
    node = root.xpath(node_path)[0]
    original_content = node.text if node.text else ""

    # Prepare prompt for CodeLlama
    prompt = f"""
    Task: Split the following XML content into four categories: task, profile, offer, and contact.
    Rules:
    1. Maintain the original wording.
    2. Do not add any new information.
    3. If a category is not applicable, omit that tag entirely.
    4. Use proper XML syntax.

    Input XML:
    <{node.tag}>
    {original_content}
    </{node.tag}>

    Output format:
    <introduction>Introduction content here</introduction>
    <task>Task content here</task>
    <profile>Profile content here</profile>
    <offer>Offer content here</offer>
    <contact>Contact content here</contact>

    Split the content:
    """

    # Get split content from CodeLlama
    split_content = ask_codellama(prompt)
    if split_content is None:
        raise Exception("Failed to get a response from CodeLlama")

    # Parse the split content
    try:
        split_root = etree.fromstring(f"<root>{split_content}</root>")
    except etree.XMLSyntaxError:
        raise Exception("CodeLlama returned invalid XML")

    # Replace original node with new nodes
    parent = node.getparent()
    index = parent.index(node)
    parent.remove(node)
    for element in split_root:
        if element.text and element.text.strip():
            parent.insert(index, element)
            index += 1

    return etree.tostring(root, pretty_print=True, encoding='unicode')

# Function to interact with CodeLlama model
def ask_codellama(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "codellama",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        if "response" in result:
            return result["response"]
        else:
            st.error(f"Unexpected response format from CodeLlama: {result}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with CodeLlama: {e}")
        return None

# Function to explain changes
def explain_changes(original_xml, new_xml):
    prompt = f"""
    Task: Explain the changes made to this XML.
    
    Before:
    {original_xml}
    
    After:
    {new_xml}
    
    Instructions:
    1. Focus on explaining which node was split.
    2. Describe how the content was distributed among the new nodes (task, profile, offer, contact).
    3. Be concise but thorough in your explanation.
    4. If a category is missing in the result, mention that it wasn't applicable.

    Explanation:
    """
    
    return ask_codellama(prompt)

# Streamlit app
def main():
    st.title("CodeLlama-based XML Node Splitter with Chroma DB")
    
    # Input XML content
    xml_content = st.text_area("Enter XML content:", height=200)
    
    # Input node path
    node_path = st.text_input("Enter node path to split (e.g., '//description'):")
    
    if st.button("Split Node"):
        if xml_content and node_path:
            try:
                # Store original XML
                original_xml = xml_content
                
                # Split the node
                new_xml = split_xml_node(xml_content, node_path)
                
                # Store result in Chroma DB
                collection.add(
                    documents=[new_xml],
                    metadatas=[{"original_xml": original_xml, "node_path": node_path}],
                    ids=[f"{node_path}_{len(collection.get()['ids'])}"]
                )
                
                # Display result
                st.subheader("Result:")
                st.code(new_xml, language="xml")
                
                # Display before and after view
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Before:")
                    st.code(original_xml, language="xml")
                with col2:
                    st.subheader("After:")
                    st.code(new_xml, language="xml")
                
                # Explain changes
                changes = explain_changes(original_xml, new_xml)
                if changes:
                    with st.expander("View Explanation", expanded=True):
                        st.subheader("Explanation of Changes:")
                        st.write(changes)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter both XML content and node path.")

    # Add a section to view previous splits
    st.subheader("Previous Splits")
    if st.button("View Previous Splits"):
        results = collection.get()
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            with st.expander(f"Split {i+1} - Node: {metadata['node_path']}"):
                st.subheader("Original:")
                st.code(metadata['original_xml'], language="xml")
                st.subheader("Split Result:")
                st.code(doc, language="xml")

if __name__ == "__main__":
    main()