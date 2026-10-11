import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import {MemoryVectorStore} from "@langchain/classic/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { RunnablePassthrough, RunnableLambda } from "@langchain/core/runnables";

async function main(){
    const loader = new TextLoader('notes.txt');

    const documents = await loader.load();

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
      Answer the question using only the context below.
     If the answer is not in the context, say "I don't know".

        Context:
       {context}

        Question:
       {question}
      `);

     const model = new ChatGroq({
      model: "openai/gpt-oss-20b",
      temperature: 0,
       });

        const ragChain = RunnablePassthrough.assign({
         context: new RunnableLambda({
       func: async ({ question }) => {
       const docs = await retriever.invoke(question);
       return docs.map((doc) => doc.pageContent).join("\n\n");
         },
         }),
          })
       .pipe(prompt)
        .pipe(model);


        const query = "what is the main topic inside notes?"
         const response = await ragChain.invoke({
          question: query,
         });

          console.log("\n--- RAG Chain Answer ---");
             console.log(response.content);

       // console.log(formattedprompt);
}
main();
