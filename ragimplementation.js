import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import {MemoryVectorStore} from "@langchain/classic/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { RunnablePassthrough, RunnableLambda } from "@langchain/core/runnables";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

async function main(){
    const loader = new PDFLoader('data/ApplicationForm.pdf');

    const documents = await loader.load();

    console.log("total pages :",documents.length);

    const textsplitter = new RecursiveCharacterTextSplitter({
        chunkSize : 200,
        chunkOverlap:20
    });

    const chunks = await textsplitter.splitDocuments(documents);
    
    // chunks.forEach((chunk,index)=>{
    //     console.log(`\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n chunk ${index} :${chunk.pageContent}`);
    // })

    const embeddings = new HuggingFaceTransformersEmbeddings({
        model: "Xenova/all-MiniLM-L6-v2",
    })

    const vectorstore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

    //ueser query
    const retriever = vectorstore.asRetriever({ k: 2 });

     const prompt = ChatPromptTemplate.fromTemplate(`
     You are a helpful assistant answering questions about a PDF.

     Use the document context and conversation history to answer.
     If the answer is not available in the document, say "I don't know".

     Conversation history:
      {chatHistory}

     Document context:
     {context}

     Current question:
      {question}
       `);

     const model = new ChatGroq({
      model: "openai/gpt-oss-20b",
      temperature: 0,
       });

       //memeory store
       const chatHistory = [];


       
const contextualizePrompt = ChatPromptTemplate.fromTemplate(`
Given the conversation history and the latest question,
rewrite the latest question as a standalone question
that can be understood without the conversation history.

Do not answer the question.
Return only the rewritten question.

Conversation history:
{chatHistory}

Latest question:
{question}
`);

const contextualizeChain = contextualizePrompt.pipe(model);





        const ragChain = RunnablePassthrough.assign({
         context: new RunnableLambda({
       func: async ({ question }) => {
      
      const docs = await retriever.invoke(question);

     console.log("\n--- Retrieved PDF Chunks ---");

      docs.forEach((doc, index) => {
        console.log(`\nChunk ${index + 1}:`);
      console.log(doc.pageContent);
        });

return docs.map((doc) => doc.pageContent).join("\n\n");

         },
         }),
          })
       .pipe(prompt)
        .pipe(model);


      
const questions = [
  "What information does this form ask for?",
  "Which details should the applicant provide?",
  "Summarize the form in one sentence."
];

for (const query of questions) {
  const response = await ragChain.invoke({
    question: query,
    chatHistory: chatHistory
      .map((msg) => `${msg.role}: ${msg.content}`)
      .join("\n"),
  });

  console.log("\n--- Question ---");
  console.log(query);

  console.log("\n--- Answer ---");
  console.log(response.content);

  chatHistory.push(
    { role: "user", content: query },
    { role: "assistant", content: response.content }
  );
}

}
main();
