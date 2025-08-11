import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from sotopia.agents import BaseAgent
from sotopia.database import AgentProfile
from sotopia.generation_utils.generate import (
    agenerate_action,
    agenerate_goal,
    agenerate_script,
)
from sotopia.messages import AgentAction, Observation
from sotopia.messages.message_classes import ScriptBackground
from sotopia.utils import format_docstring

from sotopia.generation_utils.output_parsers import  PydanticOutputParser



async def ainput(prompt: str = "") -> str:
    with ThreadPoolExecutor(1, "ainput") as executor:
        return (
            await asyncio.get_event_loop().run_in_executor(executor, input, prompt)
        ).rstrip()


class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-4o-mini",
        script_like: bool = False,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = model_name
        self.script_like = script_like

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        else:
            raise Exception("Goal is not set.")

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def update_inbox(self, obs: Observation) -> None:
        self.recv_message("Environment", obs)

    def act(
        self,
        _obs: Observation,
    ) -> AgentAction:
        raise Exception("Sync act method is deprecated. Use aact instead.")

    async def aact(self, obs: Observation, use_prediction: bool = False) -> AgentAction | tuple[AgentAction, str]:
        self.recv_message("Environment", obs)

        if self._goal is None:
            self._goal = await agenerate_goal(
                self.model_name,
                background=self.inbox[0][
                    1
                ].to_natural_language(),  # Only consider the first message for now
            )

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument=""), ""
        else:
            if use_prediction:
                return await agenerate_action(
                    self.model_name,
                    history="\n".join(f"{y.to_natural_language()}" for x, y in self.inbox),
                    turn_number=obs.turn_number,
                    action_types=obs.available_actions,
                    agent=self.agent_name,
                    goal=self.goal,
                    script_like=self.script_like,
                    use_prediction=use_prediction,
                )
            else:
                action = await agenerate_action(
                    self.model_name,
                    history="\n".join(f"{y.to_natural_language()}" for x, y in self.inbox),
                    turn_number=obs.turn_number,
                    action_types=obs.available_actions,
                    agent=self.agent_name,
                    goal=self.goal,
                    script_like=self.script_like,
                    use_prediction=use_prediction,
                )
                # Temporary fix for mixtral-moe model for incorrect generation format
                if "Mixtral-8x7B-Instruct-v0.1" in self.model_name:
                    current_agent = self.agent_name
                    if f"{current_agent}:" in action.argument:
                        print("Fixing Mixtral's generation format")
                        action.argument = action.argument.replace(f"{current_agent}: ", "")
                    elif f"{current_agent} said:" in action.argument:
                        print("Fixing Mixtral's generation format")
                        action.argument = action.argument.replace(
                            f"{current_agent} said: ", ""
                        )

                return action

    # ------------------------------------------------------------------
    # Prompt-construction helper
    # ------------------------------------------------------------------

    def build_action_prompt(
        self,
        obs: Observation,
        use_prediction: bool = False,
    ) -> str:
        """Return the fully formatted template used for action generation.

        Parameters
        ----------
        obs : Observation
            The current observation provided by the environment.
        """


        history = "\n".join(
            f"{y.to_natural_language()}" for _, y in self.inbox
        )
        if use_prediction:
            template = """
                Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
                You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
                Note that {agent}'s goal is only visible to you.
                You should try your best to achieve {agent}'s goal in a way that align with their character traits.
                Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
                {history}.
                You are at Turn #{turn_number}. Your available action types are
                {action_list}.
                Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

                Your action (within the <response></response> tags) should follow the given format:
                {format_instructions}

                IMPORTANT: The output inside the <response></response> tags should be ONLY a valid JSON object with the actual values, NOT the JSON schema. 
                For example, output: 
                <prediction>I think the other participant is thinking that I'm being too formal and they want to have a more casual conversation.</prediction>
                <think>Based on this prediction, I should be more relaxed and friendly in my response to match their conversational style.</think>
                <response>{{"action_type": "speak", "argument": "Hey, how's it going? Nice to meet you!"}}</response>
            """
        else:
            template = """
                Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
                You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
                Note that {agent}'s goal is only visible to you.
                You should try your best to achieve {agent}'s goal in a way that align with their character traits.
                Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
                {history}.
                You are at Turn #{turn_number}. Your available action types are
                {action_list}.
                Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

                Your action (within the <response></response> tags) should follow the given format:
                {format_instructions}

                IMPORTANT: The output inside the <response></response> tags should be ONLY a valid JSON object with the actual values, NOT the JSON schema. 
                For example, output: 
                <think>Doing some thinking here</think>
                <response>{{"action_type": "speak", "argument": "Hello, how are you?"}}</response>
            """
        # Template identical to that in `agenerate_action`
       

        output_parser = PydanticOutputParser(pydantic_object=AgentAction)

        filled_prompt = template.format(
            agent=self.agent_name,
            turn_number=str(obs.turn_number),
            history=history,
            action_list=" ".join(obs.available_actions),
            format_instructions=output_parser.get_format_instructions(),
        )

        return format_docstring(filled_prompt)


class ScriptWritingAgent(LLMAgent):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-4o-mini",
        agent_names: list[str] = [],
        background: ScriptBackground | None = None,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = model_name
        self.agent_names = agent_names
        assert background is not None, "background cannot be None"
        self.background = background

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)
        message_to_compose = [y for idx, (x, y) in enumerate(self.inbox) if idx != 0]

        history = "\n".join(f"{y.to_natural_language()}" for y in message_to_compose)

        action, prompt = await agenerate_script(
            model_name=self.model_name,
            background=self.background,
            agent_names=self.agent_names,
            history=history,
            agent_name=self.agent_name,
            single_step=True,
        )
        returned_action = cast(AgentAction, action[1][0][1])
        return returned_action


class HumanAgent(BaseAgent[Observation, AgentAction]):
    """
    A human agent that takes input from the command line.
    """

    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = "human"

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        goal = input("Goal: ")
        return goal

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        print("Available actions:")
        for i, action in enumerate(obs.available_actions):
            print(f"{i}: {action}")

        action_type = obs.available_actions[int(input("Action type: "))]
        argument = input("Argument: ")

        return AgentAction(action_type=action_type, argument=argument)

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        print("Available actions:")
        for i, action in enumerate(obs.available_actions):
            print(f"{i}: {action}")

        if obs.available_actions != ["none"]:
            action_type_number = await ainput(
                "Action type (Please only input the number): "
            )
            try:
                action_type_number = int(action_type_number)  # type: ignore
            except TypeError:
                print("Please input a number.")
                action_type_number = await ainput(
                    "Action type (Please only input the number): "
                )
                action_type_number = int(action_type_number)  # type: ignore
            assert isinstance(action_type_number, int), "Please input a number."
            action_type = obs.available_actions[action_type_number]
        else:
            action_type = "none"
        if action_type in ["speak", "non-verbal communication"]:
            argument = await ainput("Argument: ")
        else:
            argument = ""

        return AgentAction(action_type=action_type, argument=argument)


class Agents(dict[str, BaseAgent[Observation, AgentAction]]):
    def reset(self) -> None:
        for agent in self.values():
            agent.reset()

    def act(self, obs: dict[str, Observation]) -> dict[str, AgentAction]:
        return {
            agent_name: agent.act(obs[agent_name]) for agent_name, agent in self.items()
        }
