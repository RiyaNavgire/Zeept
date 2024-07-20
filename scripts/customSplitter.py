from langchain.text_splitter import TextSplitter
import re
from langchain.schema import Document


class customTextSplitter(TextSplitter):
    def __init__(self, delimiter="\n", section_pattern=r"^[A-Z][a-z]*\n=+\n"):
        self.delimiter = delimiter
        self.section_pattern = section_pattern

    def split_text(self, text):
        # Split the text into sections based on the pattern
        sections = re.split(self.section_pattern, text)
        
        # Re-add the section headers to their corresponding sections
        headers = re.findall(self.section_pattern, text)
        
        # Combine headers and sections
        chunks = []
        for i, section in enumerate(sections):
            if i < len(headers):
                chunks.append(headers[i] + section)
            else:
                chunks.append(section)
        
        return chunks

    def split_and_package(self, text, source ,section):
        chunks = self.split_text(text)
        documents = []

        for chunk in chunks:
            # Extract section header as metadata
            section_header = chunk.split('\n', 1)[0].strip()
            metadata = {"section": section, "source": source}
            documents.append(Document(page_content=chunk, metadata=metadata))
        return documents