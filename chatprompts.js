import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,PromptTemplate } from "@langchain/core/prompts";
import dotenv from 'dotenv';
dotenv.config();

const chatmodel = new ChatOpenAI(process.env.OPENAI_KEY);

const sys_template = "you are a senior {role}";
const human_template = "ask me about {topic}";

const prompt = ChatPromptTemplate.fromMessages(
    ["system",sys_template],
    ["human",human_template]
);

const formattedprompt = await  prompt.formatMessages({
    role:"developer",
    topic :"js",
});

const res = await chatmodel.invoke(formattedprompt);
console.log(res);

//or diff way to create system and human prompt
const sys = SystemMessagePromptTemplate.fromTemplate("you are a senior interviewer");
const hum = HumanMessagePromptTemplate.fromTemplate("what is {tetx}");

const newprompt = ChatPromptTemplate.fromMessages([sys,hum]);

const actualprompt = await newprompt.formatMessages({
    text :"python"
});
const result  = await chatmodel.invoke(actualprompt);
console.log(result);
