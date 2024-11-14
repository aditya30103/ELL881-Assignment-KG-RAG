import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Memory
import json
import openai
import os
import sys
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
from dotenv import load_dotenv, find_dotenv
import torch
from langchain import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
from kg_rag.config_loader import *
import ast
import requests
import google.generativeai as genai
import traceback
import re
import json

memory = Memory("cachegpt", verbose=0)

# Config openai library
config_file = config_data['GPT_CONFIG_FILE']
load_dotenv(config_file)
api_key = os.environ.get('API_KEY')
api_version = os.environ.get('API_VERSION')
resource_endpoint = os.environ.get('RESOURCE_ENDPOINT')
openai.api_type = config_data['GPT_API_TYPE']
openai.api_key = api_key
if resource_endpoint:
    openai.api_base = resource_endpoint
if api_version:
    openai.api_version = api_version

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


torch.cuda.empty_cache()
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def get_spoke_api_resp(base_uri, end_point, params=None):
    uri = base_uri + end_point
    if params:
        return requests.get(uri, params=params)
    else:
        return requests.get(uri)

@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def get_context_using_spoke_api(node_value):
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)
    data_spoke_types = result.json()
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]
    api_params = {
        'node_filters' : filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth' : config_data['depth']
    }
    node_type = "Disease"
    attribute = "name"
    nbr_end_point = "/api/v1/neighborhood/{}/{}/{}".format(node_type, attribute, node_value)
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    node_context = result.json()
    nbr_nodes = []
    nbr_edges = []
    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["description"]))
                else:
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["name"]))
            except:
                nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["identifier"]))
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)                    
                except:
                    try:                    
                        preprint_list = ast.literal_eval(item["data"]["properties"]["preprint_list"])
                        if len(preprint_list) > 0:                                                    
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"]["pmid_list"])
                            pmid_list = map(lambda x:"pubmedId:"+x, pmid_list)
                            if len(pmid_list) > 0:
                                provenance = ", ".join(pmid_list)
                            else:
                                provenance = "Based on data from Institute For Systems Biology (ISB)"
                    except:                                
                        provenance = "SPOKE-KG"     
            try:
                evidence = item["data"]["properties"]
            except:
                evidence = None
            nbr_edges.append((item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance, evidence))
    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])
    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    merge_1.loc[:,"node_name"] = merge_1.node_type + " " + merge_1.node_name
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1 = merge_1.rename(columns={"node_name":"source"})
    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    merge_2.loc[:,"node_name"] = merge_2.node_type + " " + merge_2.node_name
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2 = merge_2.rename(columns={"node_name":"target"})
    merge_2 = merge_2[["source", "edge_type", "target", "provenance", "evidence"]]
    merge_2.loc[:, "predicate"] = merge_2.edge_type.apply(lambda x:x.split("_")[0])
    merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + "."
    context = merge_2.context.str.cat(sep=' ')
    context += node_value + " has a " + node_context[0]["data"]["properties"]["source"] + " identifier of " + node_context[0]["data"]["properties"]["identifier"] + " and Provenance of this is from " + node_context[0]["data"]["properties"]["source"] + "."
    return context, merge_2
        
#         if edge_evidence:
#             merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + " and attributes associated with this association is in the following JSON format:\n " + merge_2.evidence.astype('str') + "\n\n"
#         else:
#             merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + ". "
#         context = merge_2.context.str.cat(sep=' ')
#         context += node_value + " has a " + node_context[0]["data"]["properties"]["source"] + " identifier of " + node_context[0]["data"]["properties"]["identifier"] + " and Provenance of this is from " + node_context[0]["data"]["properties"]["source"] + "."
#     return context



def get_prompt(instruction, new_system_prompt):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + system_prompt + instruction + E_INST
    return prompt_template

def llama_model(model_name, branch_name, cache_dir, temperature=0, top_p=1, max_new_tokens=512, stream=False, method='method-1'):
    if method == 'method-1':
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                 revision=branch_name,
                                                 cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name,                                             
                                            device_map='auto',
                                            torch_dtype=torch.float16,
                                            revision=branch_name,
                                            cache_dir=cache_dir)
    elif method == 'method-2':
        import transformers
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, 
                                                                revision=branch_name, 
                                                                cache_dir=cache_dir, 
                                                                legacy=False,
                                                                token="hf_WbtWB...")
        model = transformers.LlamaForCausalLM.from_pretrained(model_name, 
                                                              device_map='auto', 
                                                              torch_dtype=torch.float16, 
                                                              revision=branch_name, 
                                                              cache_dir=cache_dir,
                                                              token="hf_WbtWB...")        
    if not stream:
        pipe = pipeline("text-generation",
                    model = model,
                    tokenizer = tokenizer,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    max_new_tokens = max_new_tokens,
                    do_sample = True
                    )
    else:
        streamer = TextStreamer(tokenizer)
        pipe = pipeline("text-generation",
                    model = model,
                    tokenizer = tokenizer,
                    torch_dtype = torch.bfloat16,
                    device_map = "auto",
                    max_new_tokens = max_new_tokens,
                    do_sample = True,
                    streamer=streamer
                    )        
    llm = HuggingFacePipeline(pipeline = pipe,
                              model_kwargs = {"temperature":temperature, "top_p":top_p})
    return llm



@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    
    response = openai.ChatCompletion.create(
        temperature=temperature,
        # deployment_id=chat_deployment_id,
        model=chat_model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]
    )
    
    if 'choices' in response \
       and isinstance(response['choices'], list) \
       and len(response) >= 0 \
       and 'message' in response['choices'][0] \
       and 'content' in response['choices'][0]['message']:
        return response['choices'][0]['message']['content']
    else:
        return 'Unexpected response'

@memory.cache
def get_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    res = fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature)
    return res





@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def fetch_Gemini_response(instruction, system_prompt, temperature=0.0):
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_prompt,
    )
    response = model.generate_content(instruction)
    return response.text
    


@memory.cache
def get_Gemini_response(instruction, system_prompt, temperature=0.0):
    res = fetch_Gemini_response(instruction, system_prompt, temperature)
    return res




def stream_out(output):
    CHUNK_SIZE = int(round(len(output)/50))
    SLEEP_TIME = 0.1
    for i in range(0, len(output), CHUNK_SIZE):
        print(output[i:i+CHUNK_SIZE], end='')
        sys.stdout.flush()
        time.sleep(SLEEP_TIME)
    print("\n")

def get_gpt35():
    chat_model_id = 'gpt-35-turbo' if openai.api_type == 'azure' else 'gpt-3.5-turbo'
    chat_deployment_id = chat_model_id if openai.api_type == 'azure' else None
    return chat_model_id, chat_deployment_id

def get_gpt4o_mini():
    chat_model_id = 'gpt-4o-mini' if openai.api_type == 'azure' else 'gpt-4o-mini'
    chat_deployment_id = chat_model_id if openai.api_type == 'azure' else None
    return chat_model_id, chat_deployment_id

def get_gemini():
    chat_model_id = 'gemini-1.5-flash' if openai.api_type == 'azure' else 'gemini-1.5-flash'
    chat_deployment_id = chat_model_id if openai.api_type == 'azure' else None
    return chat_model_id, chat_deployment_id

def disease_entity_extractor(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    resp = get_GPT_response(text, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None
    
def disease_entity_extractor_v2(text, model_id):
    assert model_id in ("gemini-1.5-flash")
    prompt_updated = system_prompts["DISEASE_ENTITY_EXTRACTION"] + "\n" + "Sentence : " + text
    resp = get_Gemini_response(prompt_updated, system_prompts["DISEASE_ENTITY_EXTRACTION"], temperature=0.0)
    if resp.startswith("```json\n"):
        resp = resp.replace("```json\n", "", 1)
    if resp.endswith("\n```"):
        resp = resp.replace("\n```", "", -1)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None
    

def load_sentence_transformer(sentence_embedding_model):
    return SentenceTransformerEmbeddings(model_name=sentence_embedding_model)

def load_chroma(vector_db_path, sentence_embedding_model):
    embedding_function = load_sentence_transformer(sentence_embedding_model)
    return Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)

def retrieve_context(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence,model_id="gpt-3.5-turbo", api=False):
    print("question:", question)
    entities = disease_entity_extractor_v2(question, model_id)
    print("entities:", entities)
    node_hits = []
    if entities:
        max_number_of_high_similarity_context_per_node = int(context_volume/len(entities))
        for entity in entities:
            node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
            node_hits.append(node_search_result[0][0].page_content)
        question_embedding = embedding_function.embed_query(question)
        node_context_extracted = ""
        for node_name in node_hits:
            if not api:
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
            else:
                node_context,context_table = get_context_using_spoke_api(node_name)
            node_context_list = node_context.split(". ")        
            node_context_embeddings = embedding_function.embed_documents(node_context_list)
            similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
            similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
            if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
            high_similarity_context = [node_context_list[index] for index in high_similarity_indices]            
            if edge_evidence:
                high_similarity_context = list(map(lambda x:x+'.', high_similarity_context)) 
                context_table = context_table[context_table.context.isin(high_similarity_context)]
                context_table.loc[:, "context"] =  context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
                node_context_extracted += context_table.context.str.cat(sep=' ')
            else:
                node_context_extracted += ". ".join(high_similarity_context)
                node_context_extracted += ". "
        return node_context_extracted
    else:
        node_hits = vectorstore.similarity_search_with_score(question, k=5)
        max_number_of_high_similarity_context_per_node = int(context_volume/5)
        question_embedding = embedding_function.embed_query(question)
        node_context_extracted = ""
        for node in node_hits:
            node_name = node[0].page_content
            if not api:
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
            else:
                node_context, context_table = get_context_using_spoke_api(node_name)
            node_context_list = node_context.split(". ")        
            node_context_embeddings = embedding_function.embed_documents(node_context_list)
            similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
            similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
            if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
            high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
            if edge_evidence:
                high_similarity_context = list(map(lambda x:x+'.', high_similarity_context))
                context_table = context_table[context_table.context.isin(high_similarity_context)]
                context_table.loc[:, "context"] =  context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
                node_context_extracted += context_table.context.str.cat(sep=' ')
            else:
                node_context_extracted += ". ".join(high_similarity_context)
                node_context_extracted += ". "
        return node_context_extracted
    
    
def interactive(question, vectorstore, node_context_df, embedding_function_for_context_retrieval, llm_type, edge_evidence, system_prompt, api=True, llama_method="method-1"):
    print(" ")
    input("Press enter for Step 1 - Disease entity extraction using GPT-3.5-Turbo")
    print("Processing ...")
    entities = disease_entity_extractor_v2(question, "gpt-4o-mini")
    max_number_of_high_similarity_context_per_node = int(config_data["CONTEXT_VOLUME"]/len(entities))
    print("Extracted entity from the prompt = '{}'".format(", ".join(entities)))
    print(" ")
    
    input("Press enter for Step 2 - Match extracted Disease entity to SPOKE nodes")
    print("Finding vector similarity ...")
    node_hits = []
    for entity in entities:
        node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
        node_hits.append(node_search_result[0][0].page_content)
    print("Matched entities from SPOKE = '{}'".format(", ".join(node_hits)))
    print(" ")
    
    input("Press enter for Step 3 - Context extraction from SPOKE")
    node_context = []
    for node_name in node_hits:
        if not api:
            node_context.append(node_context_df[node_context_df.node_name == node_name].node_context.values[0])
        else:
            context, context_table = get_context_using_spoke_api(node_name)
            node_context.append(context)
    print("Extracted Context is : ")
    print(". ".join(node_context))
    print(" ")

    input("Press enter for Step 4 - Context pruning")
    question_embedding = embedding_function_for_context_retrieval.embed_query(question)
    node_context_extracted = ""
    for node_name in node_hits:
        if not api:
            node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
        else:
            node_context, context_table = get_context_using_spoke_api(node_name)                        
        node_context_list = node_context.split(". ")        
        node_context_embeddings = embedding_function_for_context_retrieval.embed_documents(node_context_list)
        similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
        similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
        percentile_threshold = np.percentile([s[0] for s in similarities], config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
        high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"]]
        if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
            high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
        high_similarity_context = [node_context_list[index] for index in high_similarity_indices]               
        if edge_evidence:
            high_similarity_context = list(map(lambda x:x+'.', high_similarity_context)) 
            context_table = context_table[context_table.context.isin(high_similarity_context)]
            context_table.loc[:, "context"] =  context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
            node_context_extracted += context_table.context.str.cat(sep=' ')
        else:
            node_context_extracted += ". ".join(high_similarity_context)
            node_context_extracted += ". "
    print("Pruned Context is : ")
    print(node_context_extracted)
    print(" ")
    
    input("Press enter for Step 5 - LLM prompting")
    print("Prompting ", llm_type)
    if llm_type == "llama":
        from langchain import PromptTemplate, LLMChain
        template = get_prompt("Context:\n\n{context} \n\nQuestion: {question}", system_prompt)
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm = llama_model(config_data["LLAMA_MODEL_NAME"], config_data["LLAMA_MODEL_BRANCH"], config_data["LLM_CACHE_DIR"], stream=True, method=llama_method) 
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        output = llm_chain.run(context=node_context_extracted, question=question)
    elif "gpt" in llm_type:
        enriched_prompt = "Context: "+ node_context_extracted + "\n" + "Question: " + question
        output = get_GPT_response(enriched_prompt, system_prompt, llm_type, llm_type, temperature=config_data["LLM_TEMPERATURE"])
        stream_out(output)


### MODE 1 & 3 UTILITIES

def retrieve_context_mode1(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence,model_id="gpt-3.5-turbo", api=False):
    print("question:", question)
    entities = disease_entity_extractor_v2(question, model_id)
    print("entities:", entities)
    node_hits = []
    if entities:
        max_number_of_high_similarity_context_per_node = int(context_volume/len(entities))
        for entity in entities:
            node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
            node_hits.append(node_search_result[0][0].page_content)
        question_embedding = embedding_function.embed_query(question)
        node_context_extracted = ""
        for node_name in node_hits:
            if not api:
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
            else:
                node_context,context_table = get_context_using_spoke_api(node_name)
            node_context_list = node_context.split(". ")        
            node_context_embeddings = embedding_function.embed_documents(node_context_list)
            similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
            similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
            if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
            high_similarity_context = [node_context_list[index] for index in high_similarity_indices]            
            if edge_evidence:
                high_similarity_context = list(map(lambda x:x+'.', high_similarity_context)) 
                context_table = context_table[context_table.context.isin(high_similarity_context)]
                context_table.loc[:, "context"] =  context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
                node_context_extracted += context_table.context.str.cat(sep=' ')
            else:
                node_context_extracted += ". ".join(high_similarity_context)
                node_context_extracted += ". "
        return node_context_extracted, node_hits

def normalize_text(text):
    text = text.strip().lower()
    # Replace various Unicode apostrophes with standard apostrophe
    text = re.sub(r"[‘’′`´’]", "'", text)
    # Replace various hyphens with standard hyphen
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text

def parse_context_per_disease(context, entities):
    normalized_entities = [normalize_text(disease) for disease in entities]
    disease_dict = {disease: {"associations": [], "entity_info": {}} for disease in normalized_entities}

    # Updated regular expression with known predicates
    association_pattern = re.compile(
        r'^\s*(?P<source_type>\S+)\s+'
        r'(?P<source_name>.+?)\s+'
        r'(?P<predicate>associates|resembles|treats|isa|presents|localizes)\s+'
        r'(?P<target_type>\S+)\s+'
        r'(?P<target_name>.+?)'
        r'(?:\s+and Provenance of this association is\s+(?P<provenance>.*?))?\.?$',
        re.IGNORECASE
    )

    entity_info_pattern = re.compile(
        r'^\s*(?P<entity>.+?)\s+has a\s+(?P<identifier_type>.+?)\s+identifier of\s+'
        r'(?P<identifier>.+?)\s+and Provenance of this is from\s+(?P<provenance>.+?)\.?$',
        re.IGNORECASE
    )

    sentences = re.split(r'\.\s*', context.strip())
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue  

        assoc_match = association_pattern.match(sentence)
        if assoc_match:
            source_type = assoc_match.group('source_type')
            source_name = normalize_text(assoc_match.group('source_name'))
            predicate = assoc_match.group('predicate')
            target_type = assoc_match.group('target_type')
            target_name = normalize_text(assoc_match.group('target_name'))
            provenance = assoc_match.groupdict().get('provenance', '')

            if source_name in disease_dict:
                disease_dict[source_name]["associations"].append({
                    "predicate": predicate,
                    "target": f"{target_type} {assoc_match.group('target_name')}",
                    "provenance": provenance
                })
            elif target_name in disease_dict:
                disease_dict[target_name]["associations"].append({
                    "predicate": predicate,
                    "target": f"{source_type} {assoc_match.group('source_name')}",
                    "provenance": provenance
                })
        else:
            pass

    diseases_list = []
    for disease, data in disease_dict.items():
        diseases_list.append({
            "disease": disease,
            "associations": data["associations"],
            "entity_info": data["entity_info"]
        })

    result = {"diseases": diseases_list}
    return result


### MODE 4 UTILITIES

def retrieve_context_mode4(question, vectorstore, node_context_df, model_id):
    try:
        # Extract options from the question
        options = extract_options_mode4(question)
        print("Extracted options:", options)

        # Detect the entity type based on the question
        option_entity_type = detect_option_entity_type_mode4(question)
        print("Detected option entity type:", option_entity_type)

        # Extract diseases from the question
        entities = disease_entity_extractor_v2(question, model_id=model_id)
        print("Extracted entities:", entities)

        # Match diseases in SPOKE
        node_hits = []
        if entities:
            for entity in entities:
                node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
                if node_search_result and len(node_search_result) > 0 and len(node_search_result[0]) > 0:
                    node_hits.append(node_search_result[0][0].page_content)
                else:
                    print(f"No matching node found for entity '{entity}'")
        else:
            print("No entities extracted for question:", question)
            return None, None

        print("Matched nodes:", node_hits)

        # Retrieve and process context
        context_json = {"Diseases": {}}
        for node_name in node_hits:
            disease_name = node_name
            context_json["Diseases"][disease_name] = {"Associations": []}

            node_context_str, context_table, node_data = get_context_using_spoke_api_mode4(node_name)
            if context_table is None or context_table.empty:
                print(f"No context found for node '{node_name}'")
                continue  

            context_table['source_type'] = context_table['source'].apply(lambda x: x.split(' ')[0])
            context_table['source_name'] = context_table['source'].apply(extract_node_name_mode4)
            context_table['target_type'] = context_table['target'].apply(lambda x: x.split(' ')[0])
            context_table['target_name'] = context_table['target'].apply(extract_node_name_mode4)
            context_table['predicate'] = context_table['edge_type'].apply(lambda x: x.split('_')[0])

            # Filter associations based on entity type
            associations = context_table[
                (context_table['source_type'] == option_entity_type) |
                (context_table['target_type'] == option_entity_type)
            ].copy()

            if associations.empty:
                print(f"No associations of type '{option_entity_type}' found for disease '{disease_name}'")
                continue  

            associations['entity_name'] = associations.apply(
                lambda row: row['source_name'] if row['source_type'] == option_entity_type else row['target_name'],
                axis=1
            )

            associations['entity_name'] = associations['entity_name'].str.strip().str.upper()
            options_upper = [opt.strip().upper() for opt in options]

            # Prune associations based on options
            filtered_associations = associations[associations['entity_name'].isin(options_upper)]
            print(f"Number of relevant associations for disease '{disease_name}': {len(filtered_associations)}")

            # Build associations list
            for _, entry in filtered_associations.iterrows():
                entity_name = entry['entity_name']
                provenance = entry['provenance'] if isinstance(entry['provenance'], list) else [entry['provenance']]
                association = {
                    "Entity": entity_name,
                    "EntityType": option_entity_type,
                    "Provenance": provenance
                }
                context_json["Diseases"][disease_name]["Associations"].append(association)

        # Convert to JSON string
        json_context_str = json.dumps(context_json, indent=2)
        return json_context_str, options
    except Exception as e:
        print("Error in retrieve_context_mode4:", e)
        traceback.print_exc()
        return None, None

def extract_options_mode4(question):
    match = re.search(r"Given list is:(.*)", question)
    if match:
        options_str = match.group(1)
        options_str = options_str.strip().rstrip(".")
        options = [opt.strip() for opt in options_str.split(",")]
        return options
    else:
        return []

def detect_option_entity_type_mode4(question):
    if "Gene" in question:
        return "Gene"
    elif "Variant" in question or any(
        opt.startswith("rs") for opt in extract_options_mode4(question)
    ):
        return "Variant"
    else:
        return "Gene"  

def extract_node_name_mode4(node_str):
    parts = node_str.split(" ", 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return parts[0]

def get_context_using_spoke_api_mode4(node_value):
    # This function is specific to Mode 4 and does not affect Mode 0
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)
    data_spoke_types = result.json()

    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]

    api_params = {
        'node_filters': filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth': config_data['depth']
    }

    node_type = "Disease"
    attribute = "name"
    nbr_end_point = "/api/v1/neighborhood/{}/{}/{}".format(node_type, attribute, node_value)
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    node_context = result.json()

    nbr_nodes = []
    nbr_edges = []

    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"],
                                      item["data"]["properties"]["description"]))
                else:
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"],
                                      item["data"]["properties"]["name"]))
            except:
                nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"],
                                  item["data"]["properties"]["identifier"]))
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)
                except:
                    try:
                        preprint_list = ast.literal_eval(item["data"]["properties"]["preprint_list"])
                        if len(preprint_list) > 0:
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"]["pmid_list"])
                            pmid_list = list(map(lambda x: "pubmedId:" + x, pmid_list))
                            if len(pmid_list) > 0:
                                provenance = ", ".join(pmid_list)
                            else:
                                provenance = "Based on data from Institute For Systems Biology (ISB)"
                    except:
                        provenance = "SPOKE-KG"
            try:
                evidence = item["data"]["properties"]
            except:
                evidence = None
            nbr_edges.append(
                (item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance, evidence))

    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])

    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    merge_1.loc[:, "node_name"] = merge_1.node_type + " " + merge_1.node_name
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1 = merge_1.rename(columns={"node_name": "source"})

    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    merge_2.loc[:, "node_name"] = merge_2.node_type + " " + merge_2.node_name
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2 = merge_2.rename(columns={"node_name": "target"})
    merge_2 = merge_2[["source", "edge_type", "target", "provenance", "evidence"]]
    merge_2.loc[:, "predicate"] = merge_2.edge_type.apply(lambda x: x.split("_")[0])
    merge_2.loc[:, "context"] = merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + \
        " and Provenance of this association is " + merge_2.provenance + "."
    context = merge_2.context.str.cat(sep=' ')

    node_data = None
    for item in node_context:
        if item["data"]["neo4j_type"] == node_type and item["data"]["properties"].get(attribute, "").lower() == node_value.lower():
            node_data = item
            break

    if node_data:
        try:
            source_prop = node_data["data"]["properties"]["source"]
            identifier = node_data["data"]["properties"]["identifier"]
            context += node_value + " has a " + source_prop + " identifier of " + identifier + \
                " and Provenance of this is from " + source_prop + "."
        except Exception as e:
            print(f"Could not construct identifier sentence in get_context_using_spoke_api_mode4 for node '{node_value}': {e}")
    else:
        print(f"Node data not found for node '{node_value}' in get_context_using_spoke_api_mode4.")

    return context, merge_2, node_data

### MODE 5 UTILITIES

def retrieve_context_mode5(question, vectorstore, node_context_df, embedding_function, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence, model_id):
    try:
        # Extract options from the question
        options = extract_options_mode4(question)
        print("Extracted options:", options)

        # Detect the entity type based on the question
        option_entity_type = detect_option_entity_type_mode4(question)
        print("Detected option entity type:", option_entity_type)

        # Extract diseases from the question
        entities = disease_entity_extractor_v2(question, model_id=model_id)
        print("Extracted entities:", entities)

        node_hits = []
        if entities:
            for entity in entities:
                node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
                if node_search_result and len(node_search_result) > 0 and len(node_search_result[0]) > 0:
                    node_hits.append(node_search_result[0][0].page_content)
                else:
                    print(f"No matching node found for entity '{entity}'")
        else:
            print("No entities extracted for question:", question)
            return None, None, entities  # Return entities as diseases with no associations

        print("Matched nodes:", node_hits)

        # Retrieve and process context
        context_json = {"Diseases": {}}
        diseases_with_no_associations = []
        for node_name in node_hits:
            disease_name = node_name
            context_json["Diseases"][disease_name] = {"Associations": []}

            node_context_str, context_table, node_data = get_context_using_spoke_api_mode5(node_name)
            if context_table is None or context_table.empty:
                print(f"No context found for node '{node_name}'")
                diseases_with_no_associations.append(disease_name)
                continue  # Keep the associations list empty

            context_table['source_type'] = context_table['source'].apply(lambda x: x.split(' ')[0])
            context_table['source_name'] = context_table['source'].apply(extract_node_name_mode4)
            context_table['target_type'] = context_table['target'].apply(lambda x: x.split(' ')[0])
            context_table['target_name'] = context_table['target'].apply(extract_node_name_mode4)
            context_table['predicate'] = context_table['edge_type'].apply(lambda x: x.split('_')[0])

            # Filter associations based on entity type
            associations = context_table[
                (context_table['source_type'] == option_entity_type) |
                (context_table['target_type'] == option_entity_type)
            ].copy()

            if associations.empty:
                print(f"No associations of type '{option_entity_type}' found for disease '{disease_name}'")
                diseases_with_no_associations.append(disease_name)
                continue  # Keep the associations list empty

            associations['entity_name'] = associations.apply(
                lambda row: row['source_name'] if row['source_type'] == option_entity_type else row['target_name'],
                axis=1
            )
            
            associations['entity_name'] = associations['entity_name'].str.strip().str.upper()
            options_upper = [opt.strip().upper() for opt in options]

            # Prune associations based on options
            filtered_associations = associations[associations['entity_name'].isin(options_upper)]
            print(f"Number of relevant associations for disease '{disease_name}': {len(filtered_associations)}")

            if filtered_associations.empty:
                print(f"No relevant associations found for disease '{disease_name}' after pruning.")
                diseases_with_no_associations.append(disease_name)
                continue  # Keep the associations list empty

            # Build associations list
            for _, entry in filtered_associations.iterrows():
                entity_name = entry['entity_name']
                provenance = entry['provenance'] if isinstance(entry['provenance'], list) else [entry['provenance']]
                association = {
                    "Entity": entity_name,
                    "EntityType": option_entity_type,
                    "Provenance": provenance
                }
                context_json["Diseases"][disease_name]["Associations"].append(association)

        # Convert to JSON string
        json_context_str = json.dumps(context_json, indent=2)
        return json_context_str, options, diseases_with_no_associations
    except Exception as e:
        print("Error in retrieve_context_mode5:", e)
        traceback.print_exc()
        return None, None, entities  # Return entities as diseases with no associations

def retrieve_additional_context_for_diseases(disease_names, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence, model_id="gpt-3.5-turbo"):
    additional_context = ""
    for disease_name in disease_names:
        print(f"Retrieving additional context for disease: {disease_name}")
        # Retrieve context for the specific disease
        context = retrieve_context_for_disease(
            disease_name,
            vectorstore,
            embedding_function,
            node_context_df,
            context_volume,
            context_sim_threshold,
            context_sim_min_threshold,
            edge_evidence,
            model_id
        )
        if context:
            additional_context += f"Disease: {disease_name}\nContext: {context}\n\n"
        else:
            print(f"No additional context found for disease '{disease_name}'")
    return additional_context

def retrieve_context_for_disease(disease_name, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence, model_id="gpt-3.5-turbo", api=True):
    # Similar to retrieve_context but written for a specific disease
    node_hits = [disease_name]
    max_number_of_high_similarity_context_per_node = int(context_volume/len(node_hits))
    question_embedding = embedding_function.embed_query(disease_name)
    node_context_extracted = ""
    for node_name in node_hits:
        if not api:
            node_context_series = node_context_df[node_context_df.node_name == node_name].node_context
            if not node_context_series.empty:
                node_context = node_context_series.values[0]
            else:
                print(f"No context found in node_context_df for node '{node_name}'")
                continue  
        else:
            node_context, context_table, node_data = get_context_using_spoke_api_mode5(node_name)
            if not node_context:
                print(f"No context returned from get_context_using_spoke_api_mode5 for node '{node_name}'")
                continue  
        node_context_list = node_context.split(". ")
        if not node_context_list:
            print(f"No context sentences for node '{node_name}'")
            continue  
        node_context_embeddings = embedding_function.embed_documents(node_context_list)
        similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1))[0][0] for node_context_embedding in node_context_embeddings]
        similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
        percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
        high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
        if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
            high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
        if not high_similarity_indices:
            print(f"No high similarity context found for node '{node_name}'")
            continue  
        high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
        if edge_evidence:
            high_similarity_context = list(map(lambda x:x+'.', high_similarity_context))
            context_table = context_table[context_table.context.isin(high_similarity_context)]
            context_table.loc[:, "context"] =  context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
            node_context_extracted += context_table.context.str.cat(sep=' ')
        else:
            node_context_extracted += ". ".join(high_similarity_context)
            node_context_extracted += ". "
    return node_context_extracted

def get_context_using_spoke_api_mode5(node_value):
    # This function is specific to Mode 5
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)
    data_spoke_types = result.json()

    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]

    api_params = {
        'node_filters': filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth': config_data['depth']
    }

    node_type = "Disease"
    attribute = "name"
    nbr_end_point = "/api/v1/neighborhood/{}/{}/{}".format(node_type, attribute, node_value)
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    node_context = result.json()

    nbr_nodes = []
    nbr_edges = []

    if not node_context:
        print(f"No node context found for node '{node_value}' in get_context_using_spoke_api_mode5.")
        return "", None, None

    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"],
                                      item["data"]["properties"]["description"]))
                else:
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"],
                                      item["data"]["properties"]["name"]))
            except:
                nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"],
                                  item["data"]["properties"]["identifier"]))
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)
                except:
                    try:
                        preprint_list = ast.literal_eval(item["data"]["properties"]["preprint_list"])
                        if len(preprint_list) > 0:
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"]["pmid_list"])
                            pmid_list = list(map(lambda x: "pubmedId:" + x, pmid_list))
                            if len(pmid_list) > 0:
                                provenance = ", ".join(pmid_list)
                            else:
                                provenance = "Based on data from Institute For Systems Biology (ISB)"
                    except:
                        provenance = "SPOKE-KG"
            try:
                evidence = item["data"]["properties"]
            except:
                evidence = None
            nbr_edges.append(
                (item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance, evidence))

    if not nbr_nodes or not nbr_edges:
        print(f"No nodes or edges found for node '{node_value}' in get_context_using_spoke_api_mode5.")
        return "", None, None

    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])

    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    merge_1.loc[:, "node_name"] = merge_1.node_type + " " + merge_1.node_name
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1 = merge_1.rename(columns={"node_name": "source"})

    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    merge_2.loc[:, "node_name"] = merge_2.node_type + " " + merge_2.node_name
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2 = merge_2.rename(columns={"node_name": "target"})
    merge_2 = merge_2[["source", "edge_type", "target", "provenance", "evidence"]]
    merge_2.loc[:, "predicate"] = merge_2.edge_type.apply(lambda x: x.split("_")[0])
    merge_2.loc[:, "context"] = merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + \
        " and Provenance of this association is " + merge_2.provenance + "."
    context = merge_2.context.str.cat(sep=' ')


    node_data = None
    for item in node_context:
        if item["data"]["neo4j_type"] == node_type and item["data"]["properties"].get(attribute, "").lower() == node_value.lower():
            node_data = item
            break

    if node_data:
        try:
            source_prop = node_data["data"]["properties"]["source"]
            identifier = node_data["data"]["properties"]["identifier"]
            context += node_value + " has a " + source_prop + \
                " identifier of " + identifier + \
                " and Provenance of this is from " + source_prop + "."
        except Exception as e:
            print(f"Could not construct identifier sentence in get_context_using_spoke_api_mode5 for node '{node_value}': {e}")
    else:
        print(f"Node data not found for node '{node_value}' in get_context_using_spoke_api_mode5.")

    return context, merge_2, node_data