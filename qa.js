import { openai } from "./openai.js";
import { Document } from "langchain/document";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { YoutubeLoader } from "@langchain/community/document_loaders/web/youtube";

const question = process.argv[2] || "hi";
const video = "https://youtu.be/zR_iuq2evXo?si=cG8rODgRgXOx9_Cn";
const pdfFilePath = "xbox.pdf";

const docsFromYTVideo = (video) => {
  const loader = YoutubeLoader.createFromUrl(video, {
    language: "en",
    // adds the video info to the metadata property of the langchain Document object
    // I want the QA system to cite where it got the information from - source
    addVideoInfo: true,
  });
  // this method will
  // go to youtube,
  // get the transcript and load them up,
  // convert them into (an array of) langchain documents and return that
  return loader.loadAndSplit(
    // the whole youtube transcript wouldn't fit in a prompt (tokens limit)
    // we want to split up the transcript into smaller chunks
    // when we do the search we only pick the chunk that has the information needed to answer the question
    // (closest chunk - cosine similarity) and we sent it to the prompt with the query
    new CharacterTextSplitter({
      separator: " ",
      chunkSize: 2500, // tokens per chunk
      chunkOverlap: 100, // overlap between chunks
    })
  );
};

const docsFromPDF = (path) => {
  const loader = new PDFLoader(path);
  return loader.loadAndSplit(
    new CharacterTextSplitter({
      separator: ". ", // split by sentences
      chunkSize: 2500,
      chunkOverlap: 200,
    })
  );
};

const createStore = (docs) =>
  MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings());

const loadStore = async () => {
  const videoDocs = await docsFromYTVideo(video);
  const pdfDocs = await docsFromPDF(pdfFilePath);
  console.log("videoDocs", videoDocs.slice(0, 2));
  console.log("pdfDocs", pdfDocs.slice(0, 2));
  return createStore([...videoDocs, ...pdfDocs]);
};

const query = async () => {
  const store = await loadStore();
  const results = await store.similaritySearch(question, 2);
  console.log(results);
};

query();
