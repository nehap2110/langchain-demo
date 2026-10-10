import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import {MemoryVectorStore} from "@langchain/classic/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";

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
    const query = "what is the main topic of the notes?";

    const retrieval = await vectorstore.asRetriever({
        k:2
    })

    const relevantchunks = await retrieval.invoke(query);

    // relevantchunks.forEach((chunk,index)=>{
    //     console.log(`\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n relevant chunk ${index} : ${chunk.pageContent}`);
    // })

    // console.log(documents);

    const context = relevantchunks.map((chunk)=>chunk.pageContent).join("\n");

    const prompt = ChatPromptTemplate.fromTemplate(`
        you are a helpful assistant .use the following context to answer the question at the end.
        context :{context}
        question :{question}

        if the answer is not in the context,just say "i dont know".
        `)

        const formattedprompt = await prompt.invoke({
            context,
            question:query
        });

        console.log(formattedprompt);
}
main();
