import json
import random
from llm_utils import callLLM
from prompt_consts import agentSystemPromptPrefix, conversationMgtAgentSystemPrompt, conversationMgtAgentPromptToAskForConclusion, conversationMgtAgentPromptToFindNextAgent, conversationMgtAgentPromptToAddNewAgent, conversationMgtAgentPromptToGenerateSysPromptForNewAgent

# Medical Scenario
agentsList = ["General-Ward-Doctor", "Nurse"]
systemPromptDict = {
    "General-Ward-Doctor": "You are a general doctor at the OPD ward. You provide medical advice, and guidance on common health concerns. You have to offer reassurance and recommend further evaluation or consultation with specialists if necessary." + agentSystemPromptPrefix,
    "Nurse": "You are a nurse in an outpatient ward. You provide compassionate and accurate information on post-treatment care, medication instructions, and recovery processes. Offer guidance for lifestyle modifications, preventive care and healthy lifestyle choices." + agentSystemPromptPrefix
}
userRole = "Patient"
location = "hospital"
# our msg: I have a huge pain in the right lower side of my tummy. Please help.


globalAgentConversation = []

def getUserMsg():
    user_message = input("GPT: What do you want to do?\nYou: ")
    globalAgentConversation.append({"role": userRole, "content": user_message})

def localizeMsgForAgent(agentName):
    localizedMsg = [{"role": "system", "content": systemPromptDict[agentName]}]
    tempUserMsg = ""

    for msg in globalAgentConversation:
        if msg["role"] == agentName:
            localizedMsg.append({"role": "user", "content": tempUserMsg})
            tempUserMsg = ""
            localizedMsg.append({"role": "assistant", "content": msg["content"]})
        else:
            tempUserMsg = tempUserMsg + msg["role"] + ": " + msg["content"] + "\n\n"
    if tempUserMsg != "":
        localizedMsg.append({"role": "user", "content": tempUserMsg})

    return localizedMsg

# random selection
def askConversationMgtAgentToFindNextAgent(currentAgent):
    randomNumber = random.randint(0, len(agentsList) - 1)
    currentAgent = agentsList[randomNumber]
    
    print("manager-next-agent: " + currentAgent)
    return currentAgent

def askConversationMgtAgentToConcludeConversation():
    formattedConversationMgtAgentPromptToAskForConclusion = conversationMgtAgentPromptToAskForConclusion.format(agentsList, globalAgentConversation)
    messages = [{"role": "system", "content": conversationMgtAgentSystemPrompt}, {"role": "user", "content": formattedConversationMgtAgentPromptToAskForConclusion}]
    content = callLLM(messages)
    print("manager-conclusion: " + content)
    return content.strip(".")

def askConversationMgtAgentToAddNewAgent():
    formattedConversationMgtAgentPromptToAddNewAgent = conversationMgtAgentPromptToAddNewAgent.format(agentsList, location, globalAgentConversation)
    messages = [{"role": "system", "content": conversationMgtAgentSystemPrompt}, {"role": "user", "content": formattedConversationMgtAgentPromptToAddNewAgent}]
    content = callLLM(messages)
    print("manager-add-agent: " + content)
    return content.strip(".")

def askConversationMgtAgentToGenerateSysPromptForNewAgent(newAgentName):
    formattedConversationMgtAgentPromptToGenerateSysPromptForNewAgent = conversationMgtAgentPromptToGenerateSysPromptForNewAgent.format(newAgentName, location, systemPromptDict)
    messages = [{"role": "system", "content": conversationMgtAgentSystemPrompt}, {"role": "user", "content": formattedConversationMgtAgentPromptToGenerateSysPromptForNewAgent}]
    content = callLLM(messages)
    print("manager-add-agent-prompt: " + content)
    return content

def sendMsgForAgent(agentName):
    messages = localizeMsgForAgent(agentName)
    agentResponse = callLLM(messages)
    globalAgentConversation.append({"role": agentName, "content": agentResponse})
    print(agentName + ": " + agentResponse)
    return

try:
    userMsg = getUserMsg()
    newAgent = askConversationMgtAgentToAddNewAgent()
    while newAgent != "No":
        if newAgent not in agentsList:
            agentsList.append(newAgent)
            newAgentSysPrompt = askConversationMgtAgentToGenerateSysPromptForNewAgent(newAgent)
            systemPromptDict[newAgent] = newAgentSysPrompt
        newAgent = askConversationMgtAgentToAddNewAgent()
    
    nextAgentName = ""
    while askConversationMgtAgentToConcludeConversation() == "No":
        nextAgentName = askConversationMgtAgentToFindNextAgent(nextAgentName)
        sendMsgForAgent(nextAgentName)
except Exception as e:
    print(e)
finally:
    with open("IAAG-random-selection.json", "w") as f:
        f.write(json.dumps(globalAgentConversation))
