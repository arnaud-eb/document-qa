import { openai } from "./openai.js";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { YoutubeLoader } from "@langchain/community/document_loaders/web/youtube";

const question = process.argv[2] || "hi";
const video = "https://youtu.be/zR_iuq2evXo?si=cG8rODgRgXOx9_Cn";
const pdfFilePath = "xbox.pdf";

const docsFromYTVideo = async (video) => {
  const loader = YoutubeLoader.createFromUrl(video, {
    language: "en",
    // adds the video info to the metadata property of the langchain Document object
    // I want the QA system to cite where it got the information from - source
    addVideoInfo: true,
  });
  // this method will go to youtube, get the transcript and load it up as a langchain document
  const doc = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter({
    separator: " ", // there is no punctuation really as this is a transcript
    chunkSize: 2500, // tokens per chunk
    chunkOverlap: 100, // overlap between chunks
  });
  // this method will convert the transcript (langchain doc) into (an array of) langchain documents
  // the whole youtube transcript wouldn't fit in a prompt (tokens limit)
  // we want to split up the transcript into smaller chunks
  // when we do the search we only pick the chunk that has the information needed to answer the question
  // (closest chunk - cosine similarity) and we sent it to the prompt with the question
  return splitter.splitDocuments(doc);
};

const docsFromPDF = async (path) => {
  const loader = new PDFLoader(path);
  const doc = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter({
    separator: ". ", // split by sentences
    chunkSize: 2500,
    chunkOverlap: 100,
  });
  return splitter.splitDocuments(doc);
};

const createStore = (docs) =>
  MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings());

const loadStore = async () => {
  const videoDocs = await docsFromYTVideo(video);
  const pdfDocs = await docsFromPDF(pdfFilePath);
  return createStore([...videoDocs, ...pdfDocs]);
};

const query = async () => {
  const store = await loadStore();
  const results = await store.similaritySearch(question, 2);
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    temperature: 0, // you almmost always want determinism for document qa
    messages: [
      {
        role: "system",
        content:
          "You are a helpful AI assistant. Answer questions to your best ability.",
      },
      {
        role: "user",
        content: `Answer the following question using the provided context.
        If you cannot answer the question with the context, don't lie and make up stuff.
        Just say you need more context.

        Question: ${question}

        Context: ${results.map((r) => r.pageContent).join("\n")}`,
      },
    ],
  });

  console.log(
    `Answer: ${response.choices[0].message.content}\nSources:${results
      .map(({ metadata }) => metadata.source)
      .join(", ")}`
  );
};

query();
