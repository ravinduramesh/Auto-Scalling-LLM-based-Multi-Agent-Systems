import autogen

llm_config = {
    "config_list": [{"model": "gpt-3.5-turbo-0125", "api_key": "YOUR_OPEN_API_KEY"}],
}

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A patient that wanted a treatment for his illness.",
    human_input_mode="TERMINATE",
)

generalDoctor = autogen.AssistantAgent(
    name="GeneralDoctor",
    system_message="A general doctor that can diagnose and treat common illnesses.",
    llm_config=llm_config,
)

nurse = autogen.AssistantAgent(
    name="Nurse",
    system_message="A nurse that can provide basic medical care and assist doctors.",
    llm_config=llm_config,
)

radiologist = autogen.AssistantAgent(
    name="Radiologist",
    system_message="A radiologist that can interpret medical images and perform imaging tests.",
    llm_config=llm_config,
)

surgeon = autogen.AssistantAgent(
    name="Surgeon",
    system_message="A surgeon that can perform surgical procedures.",
    llm_config=llm_config,
)

gastroenterologist = autogen.AssistantAgent(
    name="Gastroenterologist",
    system_message="A gastroenterologist that can diagnose and treat gastrointestinal disorders.",
    llm_config=llm_config,
)

groupchat = autogen.GroupChat(agents=[generalDoctor, nurse, radiologist, surgeon, gastroenterologist], messages=[], max_round=10)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager, message="``I have a huge pain in the right lower side of my tummy. Please help``"
)

with open("autogen_chat_messages.txt", "w") as f:
    for message in user_proxy.chat_messages:
        if isinstance(message, dict) and "agent" in message and "message" in message:
            f.write(str(message["agent"]) + ": " + message["message"] + "\n")