import asyncio
import platform
from typing import Any

import fire

from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team


class SpeakAloud(Action):
    PROMPT_TEMPLATE: str = """
    You are a {position} in a hospital. You are asked to discuss a patient's condition. The previous medical professional has done some tests and had this to say: 
    {context}. Now it's your turn to speak out loud.
    """
    name: str = "SpeakAloud"

    async def run(self, context: str, position: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, position=position)
        
        rsp = await self._aask(prompt)

        return rsp


class MedicalProfessional(Role):
    name: str = ""
    profile: str = ""
    opponent_name: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([SpeakAloud])
        self._watch([UserRequirement, SpeakAloud])

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  # An instance of SpeakAloud

        memories = self.get_memories()
        context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in memories)

        rsp = await todo.run(context=context, position=self.profile)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.opponent_name,
        )
        self.rc.memory.add(msg)

        return msg


async def discuss_patient(condition: str, investment: float = 3.0, n_round: int = 5):
    """Run a team of medical professionals and watch them discuss a patient's condition."""
    Nurse = MedicalProfessional(name="Nurse", profile="Nurse", opponent_name="Doctor")
    Doctor = MedicalProfessional(name="Doctor", profile="Doctor", opponent_name="Radiologist")
    # groupchat = autogen.GroupChat(agents=[generalDoctor, nurse, radiologist, surgeon, gastroenterologist], messages=[], max_round=10)
    Radiologist = MedicalProfessional(name="Radiologist", profile="Radiologist", opponent_name="Surgeon")
    Surgeon = MedicalProfessional(name="Surgeon", profile="Surgeon", opponent_name="Gastroenterologist")
    Gastroenterologist = MedicalProfessional(name="Gastroenterologist", profile="Gastroenterologist", opponent_name="Nurse")
    team = Team()
    team.hire([Nurse, Doctor, Radiologist, Surgeon, Gastroenterologist])
    team.invest(investment)
    team.run_project(condition, send_to="Nurse")  # send patient's condition to Nurse and let her speak first
    await team.run(n_round=n_round)


def main(condition: str, investment: float = 3.0, n_round: int = 10):
    """
    :param condition: Patient's condition, such as "Patient: John Doe, Age: 45, Symptoms: Fever, cough, difficulty breathing"
    :param investment: contribute a certain dollar amount to watch the discussion
    :param n_round: maximum rounds of the discussion
    :return:
    """
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(discuss_patient(condition, investment, n_round))


if __name__ == "__main__":
    fire.Fire(main)  # run as python discuss_patient.py --condition="Patient: John Doe, Age: 45, Symptoms: Fever, cough, difficulty breathing" --investment=3.0 --n_round=5