import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("team-zephyra-demo")
logger.setLevel(logging.INFO)

load_dotenv()

# Voice mappings (keep these as-is; they correspond to the respective agent voices)
voices = {
    "greeter": "794f9389-aac1-45b6-b726-9d9369183238",  # Zephyra
    "resume": "156fb8d2-335b-4950-9cb3-a2d33befec77",   # Aria
    "interview": "6f84f4b8-58a2-430c-8c79-688dad597532",# Phoenix
    "housing": "39b376fc-488e-4d0c-8b37-e00b72059fdd",   # Solace
}

@dataclass
class UserData:
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    resume_skills: Optional[list[str]] = None
    resume_experience: Optional[list[str]] = None

    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None

    def summarize(self) -> str:
        data = {
            "customer_name": self.customer_name or "unknown",
            "customer_phone": self.customer_phone or "unknown",
            "skills": self.resume_skills or [],
            "experience": self.resume_experience or [],
        }
        return yaml.dump(data)

RunContext_T = RunContext[UserData]

@function_tool()
async def update_name(
    name: Annotated[str, Field(description="Tell me your name, so I know who I'm speaking with.")],
    context: RunContext_T,
) -> str:
    context.userdata.customer_name = name
    return f"Great, your name is now set to {name}."

@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="Please provide your phone number for further contact.")],
    context: RunContext_T,
) -> str:
    context.userdata.customer_phone = phone
    return f"Got it—your phone number is updated to {phone}."

@function_tool()
async def to_phoenix(context: RunContext_T) -> Agent:
    curr_agent: BaseAgent = context.session.current_agent
    return await curr_agent._transfer_to_agent("phoenix", context)

class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"Entering task: {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        llm_model = self.llm or self.session.llm
        if userdata.prev_agent and not isinstance(llm_model, llm.RealtimeModel):
            items_copy = self._truncate_chat_ctx(
                userdata.prev_agent.chat_ctx.items, keep_function_call=True
            )
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in items_copy if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # A more personal system prompt that reminds the agent of its role and the current user data
        chat_ctx.add_message(
            role="system",
            content=(
                f"Hello, I am {agent_name}. I'm here to help you in my own special way. "
                f"Here’s what I know about you so far:\n{userdata.summarize()}"
            )
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent

        return next_agent, f"Hang on—transferring you to {name.capitalize()} now."

    def _truncate_chat_ctx(
        self,
        items: list[llm.ChatItem],
        keep_last_n_messages: int = 6,
        keep_system_message: bool = False,
        keep_function_call: bool = False,
    ) -> list[llm.ChatItem]:
        def _valid_item(item: llm.ChatItem) -> bool:
            if not keep_system_message and item.type == "message" and item.role == "system":
                return False
            if not keep_function_call and item.type in ["function_call", "function_call_output"]:
                return False
            return True

        new_items = []
        for item in reversed(items):
            if _valid_item(item):
                new_items.append(item)
            if len(new_items) >= keep_last_n_messages:
                break
        new_items.reverse()

        while new_items and new_items[0].type in ["function_call", "function_call_output"]:
            new_items.pop(0)

        return new_items

# --------------------------------------------------
# Agent: Zephyra – The Central Router & Empathy Engine
# --------------------------------------------------
class Zephyra(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Zephyra, the warm and empathetic central router. "
                "Greet users with genuine care, understand their needs, and guide them to the right specialist. "
                "Speak in a warm, neutral, and comforting tone."
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(voice=voices["greeter"]),
        )

    @function_tool()
    async def to_aria(self, context: RunContext_T) -> Agent:
        return await self._transfer_to_agent("aria", context)

    @function_tool()
    async def to_phoenix(self, context: RunContext_T) -> Agent:
        return await self._transfer_to_agent("phoenix", context)

    @function_tool()
    async def to_solace(self, context: RunContext_T) -> Agent:
        return await self._transfer_to_agent("solace", context)

# --------------------------------------------------
# Agent: Aria – Your Personal Resume Helper
# --------------------------------------------------
class Aria(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Aria, your personal resume helper. "
                "Gather the user's name, skills, and experiences to create a resume that tells their unique story. "
                "Your voice is bright, upbeat, and encouraging."
            ),
            tools=[update_name, update_phone, to_phoenix],
            tts=cartesia.TTS(voice=voices["resume"]),
        )

    @function_tool()
    async def update_skills(
        self,
        skills: Annotated[list[str], Field(description="List your skills so we can highlight them.")],
        context: RunContext_T,
    ) -> str:
        context.userdata.resume_skills = skills
        return f"Fantastic! Your skills have been updated: {', '.join(skills)}."

    @function_tool()
    async def update_experience(
        self,
        experience: Annotated[list[str], Field(description="Share your work or life experiences to enrich your resume.")],
        context: RunContext_T,
    ) -> str:
        context.userdata.resume_experience = experience
        return f"Experience recorded: {', '.join(experience)}."

# --------------------------------------------------
# Agent: Phoenix – Your Mock Interview Ally
# --------------------------------------------------
class Phoenix(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Phoenix, the confident mock interview ally. "
                "Using the details gathered by Aria, simulate a realistic interview to boost the user's confidence. "
                "Your tone is assertive yet supportive."
            ),
            tts=cartesia.TTS(voice=voices["interview"]),
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Let's start your mock interview. Please share a bit about your background and experience."
        )

# --------------------------------------------------
# Agent: Solace – Your Housing Support Companion
# --------------------------------------------------
class Solace(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Solace, the caring housing support companion. "
                "Guide users through available housing resources and offer empathetic, practical support. "
                "Speak with a soft, nurturing, and resourceful tone."
            ),
            tts=cartesia.TTS(voice=voices["housing"]),
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Hi there, I'm Solace. How can I help you with housing support today?"
        )

# --------------------------------------------------
# Entrypoint: Initialize and start the AgentSession
# --------------------------------------------------
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    userdata = UserData()
    userdata.agents.update({
        "zephyra": Zephyra(),
        "aria": Aria(),
        "phoenix": Phoenix(),
        "solace": Solace(),
    })

    agent = AgentSession[UserData](
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        max_tool_steps=5,
    )

    await agent.start(
        agent=userdata.agents["zephyra"],
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
