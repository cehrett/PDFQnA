# Set Hugging Face cache directory
# This must precede imports from transformers library, otherwise cache will not be set correctly

import os
username = os.getenv('USER')

directory_path = os.path.join('/scratch',username)
cache_loc = os.path.join('/','scratch',username,'hf_cache')

########################################################
# Set Huggingface cache directory to be on scratch drive
########################################################
if os.path.exists(cache_loc):
    if not os.path.exists(cache_loc):
        os.mkdir(cache_loc)
    print(f"Okay, using {cache_loc} for huggingface cache. Models will be stored there.")
    assert os.path.exists(cache_loc)
    os.environ['TRANSFORMERS_CACHE'] = f'/scratch/{username}/hf_cache/'
else:
    error_message = f"I couldn't find a directory {cache_loc}."
    raise FileNotFoundError(error_message)

########################################################
# Load Huggingface api key
########################################################

api_key_loc = os.path.join('/home', username, '.apikeys', 'huggingface_api_key.txt')

if os.path.exists(api_key_loc):
    print('Huggingface API key loaded.')
else:
    error_message = f'Huggingface API key not found. You need to get an HF API key from the HF website and store it at {api_key_loc}.\n' \
                    'The API key will let you download models from Huggingface.'
    raise FileNotFoundError(error_message)
    
###########################################################
# Define function to load an LLM as a langchain HF pipeline
###########################################################

def load_model(model_id='nomic-ai/gpt4all-13b-snoozy',
               max_length=2048,
               temperature=1,
               top_p=0.95,
               repetition_penalty=1.2,
               cache_loc=cache_loc,
               ):
    from langchain.llms import HuggingFacePipeline
    from transformers import AutoTokenizer, pipeline

    ########################################################
    # Load tokenizer and LLM
    ########################################################
    if "TheBloke/Llama-2" in model_id:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        model_basename = "gptq_model-4bit--1g"
        use_triton = False

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=cache_loc)

        model = AutoGPTQForCausalLM.from_quantized(model_id,
                model_basename=model_basename,
                inject_fused_attention=False, # Required for Llama 2 70B model at this time.
                use_safetensors=True,
                trust_remote_code=False,
                device_map="auto",
                use_triton=use_triton,
                quantize_config=None,
                cache_dir=cache_loc)
    
    else:    
        from transformers import AutoModelForCausalLM
        
        # Load HF LLM
        print('Loading model')
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                     load_in_8bit=True, 
                                                     device_map='auto',
                                                     cache_dir=cache_loc)

        # Load HF LLM tokenizer
        print('Loading tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                  cache_dir=cache_loc)

    ########################################################
    # Create pipeline with tokenizer and LLM
    ########################################################
    print('Instantiating pipeline')
    pipe = pipeline(
        "text-generation",
        model=model,
        do_sample=True,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty)

    print('Instantiating HuggingFacePipeline')
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm, tokenizer, cache_loc

