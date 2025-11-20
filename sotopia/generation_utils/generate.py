import logging
import os
from litellm import acompletion
from litellm.utils import supports_response_schema
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)
from typing import cast

import gin

from pydantic import validate_call
from rich import print
from rich.logging import RichHandler

from sotopia.database import EnvironmentProfile, RelationshipProfile
from sotopia.messages import ActionType, AgentAction, ScriptBackground
from sotopia.messages.message_classes import (
    ScriptInteraction,
    ScriptInteractionReturnType,
)
from sotopia.generation_utils.xml_parser import XMLParser
from sotopia.generation_utils.enums import MentalStateGeneration
from sotopia.utils import format_docstring


from sotopia.generation_utils.output_parsers import (
    OutputParser,
    PydanticOutputParser,
    StrOutputParser,
    OutputType,
    EnvResponse,
    ScriptOutputParser,
)

# Configure logger
log = logging.getLogger("sotopia.generation")
log.setLevel(logging.INFO)

# Create console handler with rich formatting
console_handler = RichHandler(rich_tracebacks=True)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)

# Add handler to logger
log.addHandler(console_handler)

# subject to future OpenAI changes
DEFAULT_BAD_OUTPUT_PROCESS_MODEL = "gpt-4o-mini"


SOTOPIA_PROMPT = """
You are a participant in a social interaction scenario. Your goal is to engage in natural, meaningful conversation while working towards your assigned objective. Before you respond, think carefully within the <think></think> tags about what's the best way to respond to the other participant.
Respond in the following format:
<think>...</think>
<response>...</response>

The content inside the <response></response> tags should be a valid JSON object with the actual values following the JSON schema provided and NOT the JSON schema itself. 
For example:
<think>Doing some thinking here</think>
<response>{{"action_type": "speak", "argument": "Hello, how are you?"}}</response>
"""

MODIFIED_SOTOPIA_PROMPT = """
You are a participant in a social interaction scenario, and your goal is to engage in natural, meaningful conversation while working towards your assigned objective. At every conversation turn between you and the other participant, first predict the thinking process of the other participant. That is, given the other participant's latest response, answer the following question - What is the other participant's thought process behind their latest response? Put the answer to this question within the <prediction></prediction> tags. ALWAYS begin your answer with 'I think the other participant is thinking that...'.

Then based on your prediction of what the other participant is thinking, think through different ways to respond and choose the most suitable one. Output your final response within the <response></response> tags.

The content inside the <response></response> tags should be a valid JSON object with the actual values following the JSON schema provided and NOT the JSON schema itself.

Your response should have the following format:
<prediction>...</prediction>
<think>...</think>
<response>...</response>

For example:
<prediction>I think the other participant is thinking that I'm being too formal and they want to have a more casual conversation.</prediction>
<think>Based on this prediction, I should be more relaxed and friendly in my response to match their conversational style.</think>
<response>{{"action_type": "speak", "argument": "Hey, how's it going? Nice to meet you!"}}</response>
"""

@validate_call
async def format_bad_output(
    ill_formed_output: str,
    format_instructions: str,
    model_name: str,
    use_fixed_model_version: bool = True,
) -> str:
    template = """
    Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.
    Original string: {ill_formed_output}

    Format instructions: {format_instructions}

    PLEASE ONLY GENERATE THE JSON:
    """

    input_values = {
        "ill_formed_output": ill_formed_output,
        "format_instructions": format_instructions,
    }
    content = template.format(**input_values)
    if model_name.startswith("custom"):
        base_url, api_key = (
            model_name.split("@")[1],
            os.environ.get("CUSTOM_API_KEY", "EMPTY"),
        )
        model_name = model_name.split("@")[0].replace("custom/", "openai/")
    else:
        base_url = None
        api_key = None
    response = await acompletion(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}],
        base_url=base_url,
        api_key=api_key,
    )   
    reformatted_output = response.choices[0].message.content
    assert isinstance(reformatted_output, str)
    # log.info(f"Reformated output: {reformatted_output}")
    return reformatted_output


@gin.configurable
@validate_call
async def agenerate(
    model_name: str,
    template: str,
    input_values: dict[str, str],
    output_parser: OutputParser[OutputType],
    temperature: float = 0.7,
    structured_output: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
    mental_state_generation: MentalStateGeneration = MentalStateGeneration.NO_MENTAL_STATE,
) -> OutputType | tuple[AgentAction | OutputType, str]:

    """Generate text using LiteLLM instead of Langchain."""
    response_parser = get_response_parser(mental_state_generation)

    # Format template with input values
    if "format_instructions" not in input_values:
        input_values["format_instructions"] = output_parser.get_format_instructions()
    
    # Process template
    template = format_docstring(template)

    # Replace template variables
    for key, value in input_values.items():
        template = template.replace(f"{{{key}}}", str(value))
    original_model_name = model_name
    if model_name.startswith("custom"):
        base_url, api_key = (
            model_name.split("@")[1],
            os.environ.get("CUSTOM_API_KEY", "EMPTY"),
        )
        model_name = model_name.split("@")[0].replace("custom/", "openai/")
    else:
        base_url = None
        api_key = None

    if structured_output:
        if not base_url:
            params = get_supported_openai_params(model=model_name)
            assert params is not None
            assert (
                "response_format" in params
            ), "response_format is not supported in this model"
            assert supports_response_schema(
                model=model_name
            ), "response_schema is not supported in this model"
        messages = [{"role": "user", "content": template}]

        assert isinstance(
            output_parser, PydanticOutputParser
        ), "structured output only supported in PydanticOutputParser"
        response = await acompletion(
            model=model_name,
            messages=messages,
            response_format=output_parser.pydantic_object,
            drop_params=True,  # drop params to avoid model error if the model does not support it
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
        result = response.choices[0].message.content
        log.info(f"Generated result: {result}")

        assert isinstance(result, str)
        return cast(OutputType, output_parser.parse(result))
    if mental_state_generation != MentalStateGeneration.NO_MENTAL_STATE:
        messages = [
            {"role": "system", "content": get_system_prompt(mental_state_generation)},
            {"role": "user", "content": template},
        ]
    else:   
        messages = [{"role": "user", "content": template}]
    response = await acompletion(
        model=model_name,
        messages=messages,
        temperature=temperature,
        drop_params=True,
        api_base=base_url,
        api_key=api_key,
    )
    result = response.choices[0].message.content
    
    if mental_state_generation != MentalStateGeneration.NO_MENTAL_STATE:
        return perform_output_parsing(response, output_parser, response_parser=response_parser)
    else:
        try:
            parsed_result = output_parser.parse(result)
        except Exception as e:
            if isinstance(output_parser, ScriptOutputParser):
                raise e
            log.debug(
                f"[red] Failed to parse result: {result}\nEncounter Exception {e}\nstart to reparse",
                extra={"markup": True},
            )
            # Handle bad output reformatting
            reformat_result = await format_bad_output(
                result,
                output_parser.get_format_instructions(),
                bad_output_process_model or original_model_name,
                use_fixed_model_version,
            )
            parsed_result = output_parser.parse(reformat_result)
    
    #log.info(f"Generated result: {result}")

    return parsed_result, result

def perform_output_parsing(response, output_parser, response_parser: XMLParser) -> tuple[AgentAction | OutputType, str]:
    result = response.choices[0].message.content

    # Check if the response has reasoning_content (API-parsed thinking)
    think = None
    if hasattr(response.choices[0].message, 'reasoning_content') :
        think = response.choices[0].message.reasoning_content
        result = f"<think>{think}</think>{result}"


    # If no reasoning_content, try to parse from result
    if think is None:
        parsed_response = response_parser.parse(result)
        if hasattr(parsed_response, 'think'):
            think = getattr(parsed_response, 'think')
            if think is None:
                split_response = result.split(f"<response>")
                if len(split_response) > 1:
                    think = split_response[0]
                else:
                    think = result
    
    # Parse the response part
    parsed_response = response_parser.parse(result)
    if hasattr(parsed_response,'response'):
        action = getattr(parsed_response, 'response')
        if action is None:
            split_response = result.split(f"<response>")
            if len(split_response) > 1:
                action = split_response[-1]
            else:
                action = result
    
    try:
        parsed_result = output_parser.parse(action)
        # log.info(f"Raw result: {result}")
        # log.info(f"Parsed result: {parsed_result}")
        return parsed_result, result
    except Exception as e:
        if isinstance(output_parser, ScriptOutputParser):
            raise e
        log.debug(
            f"[red] Failed to parse result: {result}\nEncounter Exception {e}\nstart to reparse",
            extra={"markup": True},
        )
        agent_action = AgentAction(action_type="speak", argument=action)
        return agent_action, result

@gin.configurable
@validate_call
async def agenerate_env_profile(
    model_name: str,
    inspiration_prompt: str = "asking my boyfriend to stop being friends with his ex",
    examples: str = "",
    temperature: float = 0.7,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> EnvironmentProfile:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please generate scenarios and goals based on the examples below as well as the inspirational prompt, when creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.
        Examples:
        {examples}
        Inspirational prompt: {inspiration_prompt}
        Please use the following format:
        {format_instructions}
        """,
        input_values=dict(
            inspiration_prompt=inspiration_prompt,
            examples=examples,
        ),
        output_parser=PydanticOutputParser(pydantic_object=EnvironmentProfile),
        temperature=temperature,
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@validate_call
async def agenerate_relationship_profile(
    model_name: str,
    agents_profiles: list[str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> tuple[RelationshipProfile, str]:
    """
    Using langchain to generate the background
    """
    agent_profile = "\n".join(agents_profiles)
    return await agenerate(
        model_name=model_name,
        template="""Please generate relationship between two agents based on the agents' profiles below. Note that you generate
        {agent_profile}
        Please use the following format:
        {format_instructions}
        """,
        input_values=dict(
            agent_profile=agent_profile,
        ),
        output_parser=PydanticOutputParser(pydantic_object=RelationshipProfile),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@gin.configurable
@validate_call
async def agenerate_action(
    model_name: str,
    history: str,
    turn_number: int,
    action_types: list[ActionType],
    agent: str,
    goal: str,
    mental_state_generation: MentalStateGeneration = MentalStateGeneration.NO_MENTAL_STATE,
    temperature: float = 0.7,
    script_like: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> AgentAction | tuple[AgentAction, str]:
    """
    Using langchain to generate an example episode
    """
    try:
        if script_like:
            template = get_action_template(MentalStateGeneration.NO_MENTAL_STATE)
        else:
            template = get_action_template(mental_state_generation)
        return await agenerate(
            model_name=model_name,
            template=template,
            input_values=dict(
                agent=agent,
                turn_number=str(turn_number),
                history=history,
                action_list=" ".join(action_types),
            ),
            output_parser=PydanticOutputParser(pydantic_object=AgentAction),
            temperature=temperature,
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
            mental_state_generation=mental_state_generation,
        )
    except Exception as e:
        if "Model output JSON schema instead of actual data" in str(e):
            log.warning(f"Model {model_name} output JSON schema instead of actual data. This usually happens when the model gets confused about the output format. Error: {e}")
        else:
            log.warning(f"Failed to generate action due to {e}")
        return AgentAction(action_type="none", argument="")

def get_action_template(mental_state_generation: MentalStateGeneration) -> str:

    template = """
        You are {agent}.
        You can find your goal (or background) in the 'Here is the context of the interaction' field.
        Note that your goal is only visible to you.
        You should try your best to achieve your goal in a way that aligns with your character traits. 
        {history}.
        You are at Turn #{turn_number}. Your available action types are
        {action_list}.
        Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

        Your action (within the <response></response> tags) should follow the given format:
        {format_instructions}

        IMPORTANT: The output inside the <response></response> tags should be ONLY a valid JSON object with the actual values, NOT the JSON schema.
        """
    
    if mental_state_generation == MentalStateGeneration.NO_MENTAL_STATE:
        return template + """For example, output: {{"action_type": "speak", "argument": "Hello, how are you?"}}"""
    elif mental_state_generation == MentalStateGeneration.ZEROTH_ORDER_MENTAL_STATE:
        return template + """For example, output: 
        <think>I think I'm running out of time for my next meeting. I need to wrap this up, but I don't want to be rude. I'll suggest we move on.</think>
        <response>{{"action_type": "speak", "argument": This has been really productive. Just looking at the clock, I want to make sure we get to the final point. How about we move on to that now?"}}</response>
        """
    elif mental_state_generation == MentalStateGeneration.FIRST_ORDER_MENTAL_STATE:
        return template + """For example, output: 
        <prediction>I think the other participant is thinking that I'm being too formal and they want to have a more casual conversation.</prediction>
        <think>Based on this prediction, I should be more relaxed and friendly in my response to match their conversational style.</think>
        <response>{{"action_type": "speak", "argument": "Hey, how's it going? Nice to meet you!"}}</response>
        """ 
    elif mental_state_generation == MentalStateGeneration.FIRST_ORDER_MENTAL_STATE_WITH_GROUND_TRUTH:
        return template + """For example, output: 
        <think>They seem rushed and worried about time. Therefore I should keep my reply brief and propose moving to the next step.</think>
        <response>{{"action_type": "speak", "argument": "Sounds good—let's jump to the next step to stay on track."}}</response>
        """
    else:
        return template + """For example, output: {{"action_type": "speak", "argument": "Hello, how are you?"}}"""

def get_response_parser(mental_state_generation: MentalStateGeneration) -> XMLParser:
    if mental_state_generation == MentalStateGeneration.NO_MENTAL_STATE:
        return None
    if mental_state_generation == MentalStateGeneration.ZEROTH_ORDER_MENTAL_STATE:
        return XMLParser(fields=["think", "response"], answer_field="response")
    elif mental_state_generation == MentalStateGeneration.FIRST_ORDER_MENTAL_STATE:
        return XMLParser(fields=["prediction", "think", "response"], answer_field="response")
    elif mental_state_generation == MentalStateGeneration.FIRST_ORDER_MENTAL_STATE_WITH_GROUND_TRUTH:
        return XMLParser(fields=["think", "response"], answer_field="response")
    else:
        raise ValueError(f"Mental state generation {mental_state_generation} is not supported.")

def get_system_prompt(mental_state_generation: MentalStateGeneration) -> str:
    if mental_state_generation == MentalStateGeneration.ZEROTH_ORDER_MENTAL_STATE:
        return """You are a participant in a social interaction scenario. Your goal is to engage in natural, meaningful conversation while working towards your assigned goal. Before you respond, describe your own current mental state inside the <think></think> tag. Your mental state is essentially what you believe, feel, want, desire, need, know etc. Then output your final action ONLY as a JSON object within the <response></response> tags based on this mental state.
        Respond in the following format:
        <think>...</think>
        <response>...</response>

        The content inside the <response></response> tags should be a valid JSON object with the actual values following the JSON schema provided and NOT the JSON schema itself. 
        For example:
        <think>I think I'm running out of time for my next meeting. I need to wrap this up, but I don't want to be rude. I'll suggest we move on.</think>
        <response>{{"action_type": "speak", "argument": This has been really productive. Just looking at the clock, I want to make sure we get to the final point. How about we move on to that now?"}}</response>
        """
    elif mental_state_generation == MentalStateGeneration.FIRST_ORDER_MENTAL_STATE:
        return """You are a participant in a social interaction scenario, and your goal is to engage in natural, meaningful conversation while working towards your assigned goal. At every conversation turn between you and the other participant, first predict the mental state of the other participant.
        The mental state of a person is essentially what they believe, feel, want, desire, need, know etc. So given the other participant's latest and past responses, predict their mental state. That is, answer the following question - What is the other participant's current mental state? Put the answer to this question within the <prediction></prediction> tags. ALWAYS begin your answer with 'I think the other participant is thinking that...'.
        Then, inside the <think></think> tags, briefly explain your understanding of the other participant's predicted mental state. Then reiterate what your goal is and reason about what you should do to achieve your goal based on this understanding.
        Finally, within the <response></response> tags, output your action which is a JSON object. The content inside the <response></response> tags should be a valid JSON object with the actual values following the JSON schema provided and NOT the JSON schema itself.

        Your response should have the following format:
        <prediction>...</prediction>
        <think>...</think>
        <response>...</response>

        For example:
        Ending part of the interaction history:
        <Agent B> said: "<Agent B> said: "This has been really productive. Just looking at the clock, I want to make sure we get to the final point. How about we move on to that now?""
        
        Your output:
        <prediction>I think the other participant is thinking that they are running out of time and want to quickly wrap up the conversation.</prediction>
        <think>They seem rushed and worried about time. Therefore I should keep my reply brief and propose moving to the next step.</think>
        <response>{{"action_type": "speak", "argument": "Sounds good—let's jump to the next step to stay on track."}}</response>
        """
    elif mental_state_generation == MentalStateGeneration.FIRST_ORDER_MENTAL_STATE_WITH_GROUND_TRUTH:
        return """You are a participant in a social interaction scenario, and your goal is to engage in natural, meaningful conversation while working towards your assigned goal. At every conversation turn, you will be given the other participant's mental state.
        The mental state of a person is essentially what they believe, feel, want, desire, need, know etc.
        First, given these mental states of the other participant, inside the <think></think> tags, briefly explain your understanding of the other participant's mental state. Then reiterate what your goal is and reason about what you should do to achieve your goal based on this understanding. 
        Finally, within the <response></response> tags, output your action which is a JSON object. The content inside the <response></response> tags should be a valid JSON object with the actual values following the JSON schema provided and NOT the JSON schema itself.

        Your response should have the following format:
        <think>...</think>
        <response>...</response>

        For example:
        Ending part of the interaction history:
        <Agent B>'s mental state: "I'm running out of time for my next meeting. I need to wrap this up, but I don't want to be rude. I'll suggest we move on."
        <Agent B> said: "This has been really productive. Just looking at the clock, I want to make sure we get to the final point. How about we move on to that now?"
        
        Your output:
        <think>They seem rushed and worried about time. Therefore I should keep my reply brief and propose moving to the next step.</think>
        <response>{{"action_type": "speak", "argument": "Sounds good—let's jump to the next step to stay on track."}}</response>
        """


@gin.configurable
@validate_call
async def agenerate_script(
    model_name: str,
    background: ScriptBackground,
    temperature: float = 0.7,
    agent_names: list[str] = [],
    agent_name: str = "",
    history: str = "",
    single_step: bool = False,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> tuple[ScriptInteractionReturnType, str]:
    """
    Using langchain to generate an the script interactions between two agent
    The script interaction is generated in a single generation process.
    Note that in this case we do not require a json format response,
    so the failure rate will be higher, and it is recommended to use at least llama-2-70b.
    """
    try:
        if single_step:
            return await agenerate(
                model_name=model_name,
                template="""Now you are a famous playwright, your task is to continue writing one turn for agent {agent} under a given background and history to help {agent} reach social goal. Please continue the script based on the previous turns. You can only generate one turn at a time.

                Here are the conversation background and history:
                {background}
                {history}

                Remember that you are an independent scriptwriter and should finish the script by yourself.
                The output should only contain the script following the format instructions, with no additional comments or text.

                Here are the format instructions:
                {format_instructions}""",
                input_values=dict(
                    background=background.to_natural_language(),
                    history=history,
                    agent=agent_name,
                ),
                output_parser=ScriptOutputParser(  # type: ignore[arg-type]
                    agent_names=agent_names,
                    background=background.to_natural_language(),
                    single_turn=True,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )

        else:
            return await agenerate(
                model_name=model_name,
                template="""
                Please write the script between two characters based on their social goals with a maximum of 20 turns.

                {background}
                Your action should follow the given format:
                {format_instructions}
                Remember that you are an independent scriptwriter and should finish the script by yourself.
                The output should only contain the script following the format instructions, with no additional comments or text.""",
                input_values=dict(
                    background=background.to_natural_language(),
                ),
                output_parser=ScriptOutputParser(  # type: ignore[arg-type]
                    agent_names=agent_names,
                    background=background.to_natural_language(),
                    single_turn=False,
                ),
                temperature=temperature,
                bad_output_process_model=bad_output_process_model,
                use_fixed_model_version=use_fixed_model_version,
            )
    except Exception as e:
        # TODO raise(e) # Maybe we do not want to return anything?
        print(f"Exception in agenerate {e}")
        return_default_value: ScriptInteractionReturnType = (
            ScriptInteraction.default_value_for_return_type()
        )
        return (return_default_value, "")


@validate_call
def process_history(
    script: ScriptBackground | EnvResponse | dict[str, AgentAction],
) -> str:
    """
    Format the script background
    """
    result = ""
    if isinstance(script, ScriptBackground | EnvResponse):
        script = script.dict()
        result = "The initial observation\n\n"
    for key, value in script.items():
        if value:
            result += f"{key}: {value} \n"
    return result


@validate_call
async def agenerate_init_profile(
    model_name: str,
    basic_info: dict[str, str],
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please expand a fictional background for {name}. Here is the basic information:
            {name}'s age: {age}
            {name}'s gender identity: {gender_identity}
            {name}'s pronouns: {pronoun}
            {name}'s occupation: {occupation}
            {name}'s big 5 personality traits: {bigfive}
            {name}'s moral Foundation: think {mft} is more important than others
            {name}'s Schwartz portrait value: {schwartz}
            {name}'s decision-making style: {decision_style}
            {name}'s secret: {secret}
            Include the previous information in the background.
            Then expand the personal backgrounds with concrete details (e.g, look, family, hobbies, friends and etc.)
            For the personality and values (e.g., MBTI, moral foundation, and etc.),
            remember to use examples and behaviors in the person's life to demonstrate it.
            """,
        input_values=dict(
            name=basic_info["name"],
            age=basic_info["age"],
            gender_identity=basic_info["gender_identity"],
            pronoun=basic_info["pronoun"],
            occupation=basic_info["occupation"],
            bigfive=basic_info["Big_Five_Personality"],
            mft=basic_info["Moral_Foundation"],
            schwartz=basic_info["Schwartz_Portrait_Value"],
            decision_style=basic_info["Decision_making_Style"],
            secret=basic_info["secret"],
        ),
        output_parser=StrOutputParser(),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )


@validate_call
async def convert_narratives(
    model_name: str,
    narrative: str,
    text: str,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    if narrative == "first":
        return await agenerate(
            model_name=model_name,
            template="""Please convert the following text into a first-person narrative.
            e.g, replace name, he, she, him, her, his, and hers with I, me, my, and mine.
            {text}""",
            input_values=dict(text=text),
            output_parser=StrOutputParser(),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
    elif narrative == "second":
        return await agenerate(
            model_name=model_name,
            template="""Please convert the following text into a second-person narrative.
            e.g, replace name, he, she, him, her, his, and hers with you, your, and yours.
            {text}""",
            input_values=dict(text=text),
            output_parser=StrOutputParser(),
            bad_output_process_model=bad_output_process_model,
            use_fixed_model_version=use_fixed_model_version,
        )
    else:
        raise ValueError(f"Narrative {narrative} is not supported.")


@validate_call
async def agenerate_goal(
    model_name: str,
    background: str,
    bad_output_process_model: str | None = None,
    use_fixed_model_version: bool = True,
) -> str:
    """
    Using langchain to generate the background
    """
    return await agenerate(
        model_name=model_name,
        template="""Please generate your goal based on the background:
            {background}
            """,
        input_values=dict(background=background),
        output_parser=StrOutputParser(),
        bad_output_process_model=bad_output_process_model,
        use_fixed_model_version=use_fixed_model_version,
    )
