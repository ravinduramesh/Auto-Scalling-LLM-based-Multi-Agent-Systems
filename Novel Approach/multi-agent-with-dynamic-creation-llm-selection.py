import json

from openai import OpenAI

client = OpenAI(api_key='api-key')

agentSystemPromptPrefix = "Based on the given conversations, give your ideas to add new information and insight as a single dialog without mentioning your role. Make your dialog shorter and effective without repeating the information that already given in the converstaion."

# Medical Scenario
agentsList = ["General-Ward-Doctor", "Nurse"]
systemPromptDict = {
    "General-Ward-Doctor": "You are a general doctor at the OPD ward. You provide medical advice, and guidance on common health concerns. You have to offer reassurance and recommend further evaluation or consultation with specialists if necessary." + agentSystemPromptPrefix,
    "Nurse": "You are a nurse in an outpatient ward. You provide compassionate and accurate information on post-treatment care, medication instructions, and recovery processes. Offer guidance for lifestyle modifications, preventive care and healthy lifestyle choices." + agentSystemPromptPrefix
}
userRole = "Patient"
location = "hospital"
# our msg: I have a huge pain in the right lower side of my tummy. Please help.


conversationMgtAgentSystemPrompt = "You are the manager of a conversation that happens between several LLM agents. According to the given conversation between agents and the given conversation management problem, help user to manage that conversation."
conversationMgtAgentPromptToAskForConclusion = "We have agents: {} available. The following conversation has happened among them. The conversation is formatted as objects that contain the role of the speaker and content of the speech. Make sure all agent roles talk at least once. Conclude conversation only if agents thanking each other or agents repeat similar dialogs again. Is this conversation comes to a conclusion? Give only 'Yes' or 'No' without any reasons. \n\n {}"
conversationMgtAgentPromptToFindNextAgent = "We have agents: {} available. The following conversation has happened among them. The conversation is formatted as objects that contain the role of the speaker and content of the speech. Which agent should talk next? Give only the agent name by one word without any explanation. For example, if you want the backend developer to talk, just type 'Backend-Developer'. Don't let the same agent to talk consecutively. \n\n {}"

conversationMgtAgentPromptToAddNewAgent = "We have available agents: {}. The following conversation has happened among them. The conversation is formatted as objects that contain the role of the speaker and content of the speech. Suggest another LLM agent that should be in a {} to add a value to this conversation? Please give only one agent with its name by one word without any explanation. For exmaple, if you want to add a backend developer to the conversation, just type 'Backend-Developer'. Don't give agent names that are already available. If you don't need more LLM agents to be added, say 'No'. \n\n {}"
conversationMgtAgentPromptToGenerateSysPromptForNewAgent = "Generate a system message for an LLM agent that act as a {} at a {}. Give only the system prompt without any additional texts. The following is a sample set of agents and their system prompts. \n\n {}'"

globalAgentConversation = []


def getUserMsg():
    user_message = input("GPT: What do you want to do?\nYou: ")
    globalAgentConversation.append({"role": userRole, "content": user_message})

def callOpenAI(messages):
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        response_format={ "type": "text"}
    )

    content = response.choices[0].message.content.strip("'")
    return content

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

def askConversationMgtAgentToFindNextAgent():
    formattedConversationMgtAgentPromptToFindNextAgent = conversationMgtAgentPromptToFindNextAgent.format(agentsList, globalAgentConversation)
    messages = [{"role": "system", "content": conversationMgtAgentSystemPrompt}, {"role": "user", "content": formattedConversationMgtAgentPromptToFindNextAgent}]
    content = callOpenAI(messages)
    print("manager-next-agent: " + content)
    return content

def askConversationMgtAgentToConcludeConversation():
    formattedConversationMgtAgentPromptToAskForConclusion = conversationMgtAgentPromptToAskForConclusion.format(agentsList, globalAgentConversation)
    messages = [{"role": "system", "content": conversationMgtAgentSystemPrompt}, {"role": "user", "content": formattedConversationMgtAgentPromptToAskForConclusion}]
    content = callOpenAI(messages)
    print("manager-conclusion: " + content)
    return content

def askConversationMgtAgentToAddNewAgent():
    formattedConversationMgtAgentPromptToAddNewAgent = conversationMgtAgentPromptToAddNewAgent.format(agentsList, location, globalAgentConversation)
    messages = [{"role": "system", "content": conversationMgtAgentSystemPrompt}, {"role": "user", "content": formattedConversationMgtAgentPromptToAddNewAgent}]
    content = callOpenAI(messages)
    print("manager-add-agent: " + content)
    return content

def askConversationMgtAgentToGenerateSysPromptForNewAgent(newAgentName):
    formattedConversationMgtAgentPromptToGenerateSysPromptForNewAgent = conversationMgtAgentPromptToGenerateSysPromptForNewAgent.format(newAgentName, location, systemPromptDict)
    messages = [{"role": "system", "content": conversationMgtAgentSystemPrompt}, {"role": "user", "content": formattedConversationMgtAgentPromptToGenerateSysPromptForNewAgent}]
    content = callOpenAI(messages) + agentSystemPromptPrefix
    print("manager-add-agent-prompt: " + content)
    return content

def sendMsgForAgent(agentName):
    messages = localizeMsgForAgent(agentName)
    agentResponse = callOpenAI(messages)
    globalAgentConversation.append({"role": agentName, "content": agentResponse})
    print(agentName + ": " + agentResponse)
    return

try:
    userMsg = getUserMsg()
    while askConversationMgtAgentToConcludeConversation() == "No":
        newAgent = askConversationMgtAgentToAddNewAgent()
        if newAgent != "No" and (newAgent not in agentsList):
            agentsList.append(newAgent)
            newAgentSysPrompt = askConversationMgtAgentToGenerateSysPromptForNewAgent(newAgent)
            systemPromptDict[newAgent] = newAgentSysPrompt
        nextAgentName = askConversationMgtAgentToFindNextAgent()
        sendMsgForAgent(nextAgentName)
except Exception as e:
    print(e)
finally:
    with open("messages-dynamic-agent-creation-llm-selection.json", "w") as f:
        f.write(json.dumps(globalAgentConversation))
