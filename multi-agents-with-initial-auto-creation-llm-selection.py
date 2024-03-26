import json

from openai import OpenAI

client = OpenAI(api_key='api-key')

agentSystemPromptPrefix = "Based on the given conversations, give your ideas to add more information and insight as a single dialog. Make your dialog shorter and to the point without repeating the information that already given in the converstaion."

# Medical Scenario
agentsList = ["General-Ward-Doctor", "Nurse"]
systemPromptDict = {
    "General-Ward-Doctor": "You are a general doctor at the OPD ward who provides medical advice, and guidance on common health concerns. You have to offer reassurance and recommend further evaluation or consultation with specialists if necessary." + agentSystemPromptPrefix,
    "Nurse": "You are a nurse in an outpatient ward who provide compassionate and accurate information on post-treatment care, medication instructions, and recovery processes. Offer guidance for lifestyle modifications, preventive care and healthy lifestyle choices." + agentSystemPromptPrefix
}
userRole = "Patient"
location = "hospital"
# our msg: I have a huge pain in the right lower side of my tummy. Please help.


godBotSystemPrompt = "You are the manager of a conversation that happens between several LLM agents. According to the given conversation between agents and the given conversation management problem, help user to manage that conversation."
godBotPromptToAskForConclusion = "We have agents: {} available. The following conversation has happened among them. The conversation is formatted as objects that contain the role of the speaker and content of the speech. Make sure all agent roles talk at least once. Conclude conversation only if agents thanking each other or agents repeat similar dialogs again. Is this conversation comes to a conclusion? Give only 'Yes' or 'No' without any reasons. \n\n {}"
godBotPromptToFindNextAgent = "We have agents: {} available. The following conversation has happened among them. The conversation is formatted as objects that contain the role of the speaker and content of the speech. Which agent should talk next? Give only the agent name by one word without any explanation. For example, if you want the backend developer to talk, just type 'Backend-Developer'. Don't let the same agent to talk consecutively. \n\n {}"

godBotPromptToAddNewAgent = "We have available agents: {}. The following conversation has happened among them. The conversation is formatted as objects that contain the role of the speaker and content of the speech. Suggest another LLM agent that should be in a {} to add a value to this conversation? Please give only one agent with its name by one word without any explanation. For exmaple, if you want to add a backend developer to the conversation, just type 'Backend-Developer'. Don't give agent names that are already available. If you don't need more LLM agents to be added, say 'No'. \n\n {}"
godBotPromptToGenerateSysPromptForNewAgent = "Generate a system message for an LLM agent that act as a {} at a {}. As an example, if you asked to generate a system message for a backend developer at a software engineering company, you can say 'You are a backend developer at software company who is an expert of nodejs and google cloud platform.'"

globalAgentConversation = []


def getUserMsg():
    user_message = input("GPT: What do you want to do?\nYou: ")
    globalAgentConversation.append({"role": userRole, "content": user_message})

def callOpenAI(messages):
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4-turbo-preview",
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

def askGodBotToFindNextAgent():
    formattedGodBotPromptToFindNextAgent = godBotPromptToFindNextAgent.format(agentsList, globalAgentConversation)
    messages = [{"role": "system", "content": godBotSystemPrompt}, {"role": "user", "content": formattedGodBotPromptToFindNextAgent}]
    content = callOpenAI(messages)
    print("god-next-agent: " + content)
    return content

def askGodBotToConcludeConversation():
    formattedGodBotPromptToAskForConclusion = godBotPromptToAskForConclusion.format(agentsList, globalAgentConversation)
    messages = [{"role": "system", "content": godBotSystemPrompt}, {"role": "user", "content": formattedGodBotPromptToAskForConclusion}]
    content = callOpenAI(messages)
    print("god-conclusion: " + content)
    return content

def askGodBotToAddNewAgent():
    formattedGodBotPromptToAddNewAgent = godBotPromptToAddNewAgent.format(agentsList, location, globalAgentConversation)
    messages = [{"role": "system", "content": godBotSystemPrompt}, {"role": "user", "content": formattedGodBotPromptToAddNewAgent}]
    content = callOpenAI(messages)
    print("god-add-agent: " + content)
    return content

def askGodBotToGenerateSysPromptForNewAgent(newAgentName):
    formattedGodBotPromptToGenerateSysPromptForNewAgent = godBotPromptToGenerateSysPromptForNewAgent.format(newAgentName, location, globalAgentConversation)
    messages = [{"role": "system", "content": godBotSystemPrompt}, {"role": "user", "content": formattedGodBotPromptToGenerateSysPromptForNewAgent}]
    content = callOpenAI(messages) + agentSystemPromptPrefix
    print("god-add-agent-prompt: " + content)
    return content

def sendMsgForAgent(agentName):
    messages = localizeMsgForAgent(agentName)
    agentResponse = callOpenAI(messages)
    globalAgentConversation.append({"role": agentName, "content": agentResponse})
    print(agentName + ": " + agentResponse)
    return

try:
    userMsg = getUserMsg()
    newAgent = askGodBotToAddNewAgent()
    while newAgent != "No":
        if newAgent not in agentsList:
            agentsList.append(newAgent)
            newAgentSysPrompt = askGodBotToGenerateSysPromptForNewAgent(newAgent)
            systemPromptDict[newAgent] = newAgentSysPrompt
        newAgent = askGodBotToAddNewAgent()
    
    while askGodBotToConcludeConversation() == "No":
        nextAgentName = askGodBotToFindNextAgent()
        sendMsgForAgent(nextAgentName)
except Exception as e:
    print(e)
finally:
    with open("messages-initial-auto-creation-agent-llm-selection.json", "w") as f:
        f.write(json.dumps(globalAgentConversation))
