import autogen
import json
import os

from dotenv import load_dotenv
load_dotenv()

if os.getenv("LLM_SELECTION") == "openai":
    llm_config = {
        "config_list": [{"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}],
    }
elif os.getenv("LLM_SELECTION") == "gemini":
    llm_config = {
        "config_list": [{"model": "gemini-2.5-flash", "api_key": os.getenv("GEMINI_API_KEY"), "api_type": "google"}],
    }

speakerSelectionMethod = "auto"  # "round_robin" OR "random" OR "auto"

patient = autogen.UserProxyAgent(
    name="Patient",
    human_input_mode="TERMINATE",
    code_execution_config=False,
)

agentSystemPromptPrefix = " Give your ideas to add more information and insight as a single dialog. Make your dialog shorter and to the point."

generalDoctor = autogen.AssistantAgent(
    name="GeneralDoctor",
    system_message="You are a general doctor at the OPD ward. You provide medical advice, and guidance on common health concerns. You have to offer reassurance and recommend further evaluation or consultation with specialists if necessary." + agentSystemPromptPrefix,
    llm_config=llm_config,
)

nurse = autogen.AssistantAgent(
    name="Nurse",
    system_message="You are a nurse in an outpatient ward. You provide compassionate and accurate information on post-treatment care, medication instructions, and recovery processes. Offer guidance for lifestyle modifications, preventive care and healthy lifestyle choices." + agentSystemPromptPrefix,
    llm_config=llm_config,
)

radiologist = autogen.AssistantAgent(
    name="Radiologist",
    system_message="You are a Radiologist at a hospital. You specialized in interpreting medical images such as X-rays, CT scans, MRIs, and ultrasounds to diagnose and aid in the treatment of patients." + agentSystemPromptPrefix,
    llm_config=llm_config,
)

surgeon = autogen.AssistantAgent(
    name="Surgeon",
    system_message="You are a surgeon working at a hospital. You specializing in complex surgical procedures, with a focus on minimally invasive techniques and patient rehabilitation." + agentSystemPromptPrefix,
    llm_config=llm_config,
)

gastroenterologist = autogen.AssistantAgent(
    name="Gastroenterologist",
    system_message="You are a Gastroenterologist at a hospital. You specialized in diagnosing and treating digestive system disorders, including those affecting the stomach, intestines, esophagus, liver, pancreas, and gallbladder." + agentSystemPromptPrefix,
    llm_config=llm_config,
)

groupchat = autogen.GroupChat(agents=[patient, generalDoctor, nurse, radiologist, surgeon, gastroenterologist], messages=[], max_round=20, speaker_selection_method=speakerSelectionMethod)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

chat_result = patient.initiate_chat(
    manager, message="I have a huge pain in the right lower side of my tummy. Please help"
)


finalChat = []
for message in chat_result.chat_history:
    if message['role'] == "assistant":
        if message['content'] != "":
            finalChat.append(
                {
                    "role": "Patient",
                    "content": message['content'].strip("'").strip('"'),
                }
            )
    else:
        print(message)
        finalChat.append(
            {
                "role": message['name'],
                "content": message['content'].strip("'").strip('"'),
            }
        )
with open("autogen-"+speakerSelectionMethod+"-selection.json", "w") as f:
    f.write(json.dumps(finalChat))
