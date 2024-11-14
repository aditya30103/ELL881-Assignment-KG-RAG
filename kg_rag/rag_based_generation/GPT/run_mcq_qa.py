'''
This script takes the MCQ style questions from the csv file and save the result as another csv file. 
Before running this script, make sure to configure the filepaths in config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
'''

from kg_rag.utility import *
import sys


from tqdm import tqdm
CHAT_MODEL_ID = sys.argv[1]

QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]
SYSTEM_PROMPT_JSON = system_prompts["SYSTEM_PROMPT_JSON"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]


CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_kg_rag_based_mcq_{mode}.csv"


vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False


MODE = "4"
### MODE 0: Original KG_RAG                     ### 
### MODE 1: jsonlize the context from KG search ### 
### MODE 2: Add the prior domain knowledge      ### 
### MODE 3: Combine MODE 1 & 2                  ###
### MODE 4: Alternate Pipeline                  ###
### MODE 5: MODE 4 with granular fallback       ### 

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []
    
    for index, row in tqdm(question_df.iterrows(), total=306):
        try: 
            question = row["text"]
            if MODE == "0":
                ### MODE 0: Original KG_RAG  
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                enriched_prompt = "Context: "+ context + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "1":
                ### MODE 1: jsonlize the context from KG search  
                context, entities = retrieve_context_mode1(
                    question,
                    vectorstore,
                    embedding_function_for_context_retrieval,
                    node_context_df,
                    CONTEXT_VOLUME,
                    QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY,
                    edge_evidence,
                    model_id=CHAT_MODEL_ID
                )
                
                # Parse the final context to structured JSON
                parsed_json = parse_context_per_disease(context, entities)
                
                # Prepare prompt for LLM with structured JSON
                enriched_prompt = (
                    f"Context:\n{parsed_json}\n\n"
                    f"Question: {question}"
                )
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "2":
                ### MODE 2: Add the prior domain knowledge
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                prior_knowledge = system_prompts["PRIOR_KNOWLEDGE"]
                enriched_prompt = "Context: "+ context + "\n" + prior_knowledge + "\n" + "Question: "+ question
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
            
            if MODE == "3":
                ### MODE 3: Combine MODE 1 & 2          
                json_prior_knowledge = system_prompts["JSON_PRIOR_KNOWLEDGE"]        
                context, entities = retrieve_context_mode1(
                    question,
                    vectorstore,
                    embedding_function_for_context_retrieval,
                    node_context_df,
                    CONTEXT_VOLUME,
                    QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY,
                    edge_evidence,
                    model_id=CHAT_MODEL_ID
                )
                
                # Parse the final context to structured JSON
                parsed_json = parse_context_per_disease(context, entities)
                
                # Prepare prompt for LLM with structured JSON
                enriched_prompt = (
                    f"Context:\n{parsed_json}\n\n"
                    f"{json_prior_knowledge}\n\n"
                    f"Question: {question}"
                )
                
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
            
            if MODE == "4":
                ### MODE 4: Alternate Pipeline without fallback
                json_context, options = retrieve_context_mode4(
                    question,
                    vectorstore,
                    node_context_df,
                    model_id=CHAT_MODEL_ID
                )
                
                # Create the enriched prompt
                enriched_prompt = f"Context:\n{json_context}\n\nQuestion: {question}"
                
                # Get the LLM response
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT_JSON, temperature=TEMPERATURE)

            if MODE == '5':
                ### MODE 5: Mode 4 with granular fallback
                json_context, options, diseases_with_no_associations = retrieve_context_mode5(
                    question,
                    vectorstore,
                    node_context_df,
                    embedding_function_for_context_retrieval,
                    CONTEXT_VOLUME,
                    QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY,
                    edge_evidence,
                    model_id=CHAT_MODEL_ID
                )

                # Check if both JSON context and additional context are empty
                if not json_context or (diseases_with_no_associations and len(diseases_with_no_associations) == len(json.loads(json_context).get("Diseases", {}))):
                    print("No context found for any diseases. Proceeding to prompt the question directly.")
                    enriched_prompt = f"Question: {question}"
                    output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
                else:
                    additional_context = ""
                    if diseases_with_no_associations:
                        print("Diseases with no relevant associations after pruning:", diseases_with_no_associations)
                        # Retrieve additional context for these diseases
                        additional_context = retrieve_additional_context_for_diseases(
                            diseases_with_no_associations,
                            vectorstore,
                            embedding_function_for_context_retrieval,
                            node_context_df,
                            CONTEXT_VOLUME,
                            QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                            QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY,
                            edge_evidence,
                            model_id=CHAT_MODEL_ID
                        )
                    # Construct the enriched prompt
                    if additional_context.strip() == "":
                        enriched_prompt = f"Context:\n{json_context}\n\nQuestion: {question}"
                    else:
                        enriched_prompt = f"Context:\n{json_context}\n\n{additional_context}\n\nQuestion: {question}"
                    output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)


            answer_list.append((row["text"], row["correct_node"], output))
        except Exception as e:
            print("Error in processing question: ", row["text"])
            print("Error: ", e)
            answer_list.append((row["text"], row["correct_node"], "Error"))


    answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
    output_file = os.path.join(SAVE_PATH, f"{save_name}".format(mode=MODE),)
    answer_df.to_csv(output_file, index=False, header=True) 
    print("Save the model outputs in ", output_file)
    print("Completed in {} min".format((time.time()-start_time)/60))

        
        
if __name__ == "__main__":
    main()


