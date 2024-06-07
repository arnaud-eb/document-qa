import { openai } from "./openai.js";
import { Document } from "langchain/document";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { YoutubeLoader } from "@langchain/community/document_loaders/web/youtube";

const question = process.argv[2] || "hi";
const video = "https://youtu.be/zR_iuq2evXo?si=cG8rODgRgXOx9_Cn";

const createStore = (docs) =>
  MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings());

const docsFromYTVideo = (video) => {
  const loader = YoutubeLoader.createFromUrl(video, {
    language: "en",
    // adds the video info to the metadata property of the langchain Document object
    // I want the QA system to cite where it got the information from - source
    addVideoInfo: true,
  });
};
