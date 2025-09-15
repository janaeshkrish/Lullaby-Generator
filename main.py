from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv


import streamlit as st


# Load .env file
load_dotenv(find_dotenv())

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


def generate_lullaby(location, name, language):

    template = """ 
        As a children's lullaby writer, create a soothing and imaginative lullaby for a child named {name}. The lullaby should incorporate elements of {location}'s culture, nature, and traditions to make it more relatable and comforting for the child. Ensure the lullaby has a gentle rhythm and calming words to help the child relax and fall asleep peacefully. 
        
        please come up with a simple and short (70 words)
    """

    prompt = PromptTemplate(
        input_variables=["location", "name"],
        template=template,
    )

    chain_story = LLMChain(llm=llm, prompt=prompt, verbose=True, output_key="story")

    # chain to translate the story
    template_translate = """
        Can you translate the following story into {language}?
        {story}
    """

    prompt_translate = PromptTemplate(
        input_variables=["language", "story"],
        template=template_translate,
    )

    chain_translate = LLMChain(
        llm=llm, prompt=prompt_translate, verbose=True, output_key="translation"
    )

    overall_chain = SequentialChain(
        chains=[chain_story, chain_translate],
        input_variables=["location", "name", "language"],
        output_variables=["story", "translation"],
        verbose=True,
    )

    response = overall_chain({"location": location, "name": name, "language": language})

    return response


def streamlit_app():
    st.set_page_config(
        page_title="Lullaby Generator", page_icon="🎵", layout="centered"
    )

    st.title("Let AI write and translate a Lullaby for your child!  🎵")

    st.header("Enter the details below to generate a personalized lullaby:")

    location = st.text_input(label="Location (e.g., India, USA, Japan):")

    name = st.text_input(label="Character Name:")

    language = st.text_input(label="Translate to (e.g., English, Spanish, French):")

    submit_button = st.button(label="Generate Lullaby")

    if location and name and language and submit_button:
        with st.spinner("Generating lullaby..."):
            lullaby = generate_lullaby(location, name, language)

            with st.expander("English Lullaby version"):
                st.write(lullaby["story"])
            with st.expander(f"Lullaby in {language}"):
                st.write(lullaby["translation"])

            st.success("Lullaby generated successfully!")


if __name__ == "__main__":
    streamlit_app()
