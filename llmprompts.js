import {OpenAI} from '@langchain/openai';
import {PromptTemplate} from '@langchain/core/prompts';
import dotenv from "dotenv";
dotenv.config();

const model = new OpenAI({
    openAIApiKey : process.env.OPENAI_KEY
});


//prompt having no input variable

const noinputprompt = new PromptTemplate({
    inputVariables:[],
    template:"what is js?",
});

const formatnoinputprompt = await noinputprompt.format();
console.log("no input prompt :",formatnoinputprompt);

const response = await model.invoke(formatnoinputprompt);
console.log(formatnoinputprompt);

// prompt having input varaibales

const inputprompt = new PromptTemplate({
    inputVariables :[question],
    template :"what is {question}",
})

const formatinputprompt = await inputprompt.format({
    question :"what is js?",
})
console.log("prompt having input : ",formatinputprompt);

//another way to create a prompt
const prompttemplate = new PromptTemplate.fromTemplate("what is {question} and {this}?");

console.log("input variables of prompt :",prompttemplate.inputVariables);

