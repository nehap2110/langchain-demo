import { OpenAI } from "@langchain/openai";
import dotenv from 'dotenv';

dotenv.config();

//--------------- use open ai ------------------

const model = new OpenAI({
  openAIApiKey:process.env.OPENAI_KEY
});

const response = await model.invoke("who is the president of india");

console.log(response);

//--------use chat model------------------

import { ChatOpenAI } from "@langchain/openai";

const chatmodel = new ChatOpenAI({
  openAIApiKey : process.env.OPENAI_KEY,
});

const res = await chatmodel.invoke("what is the amount of water in earth?");

console.log(res);

//-------we can also define the role and messages behaviour of a system---------

import { HumanMessage,SystemMessage } from "@langchain/core/messages";

const systemmsg = new SystemMessage("you are a senior developer");
const humanmsg = new HumanMessage("What is javascript?");

const messages = [systemmsg,humanmsg];

const result = await chatmodel.invoke(messages);
console.log(result);