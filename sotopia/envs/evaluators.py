import abc
import logging
from collections import defaultdict
from typing import Generic, TypeVar

import gin
from pydantic import BaseModel, ValidationError, validate_call

from sotopia.generation_utils import PydanticOutputParser, agenerate
from sotopia.messages import (
    AgentAction,
    Message,
    ScriptEnvironmentResponse,
)
from sotopia.database.evaluation_dimensions import GoalDimension, GoalDimensionDiscrete

log = logging.getLogger("evaluators")

T_eval_dim = TypeVar("T_eval_dim", bound=BaseModel)


class EvaluationForTwoAgents(BaseModel, Generic[T_eval_dim]):
    agent_1_evaluation: T_eval_dim
    agent_2_evaluation: T_eval_dim


class Evaluator(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError


class RuleBasedTerminatedEvaluator(Evaluator):
    def __init__(self, max_turn_number: int = 20, max_stale_turn: int = 2) -> None:
        self.max_turn_number = max_turn_number
        self.max_stale_turn = max_stale_turn

    @validate_call
    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        # Rule 1: If the conversation is too long, terminate the conversation
        conversation_too_long = turn_number >= self.max_turn_number
        # Rule 2: If one of the players leaves, terminate the conversation
        p1_leaving = (
            len(messages) > 1
            and isinstance(messages[-2][1], AgentAction)
            and messages[-2][1].action_type == "leave"
        )
        p2_leaving = (
            bool(len(messages))
            and isinstance(messages[-1][1], AgentAction)
            and messages[-1][1].action_type == "leave"
        )
        # Rule 3: If the conversation is stale for too long, terminate the conversation
        stale_count = 0
        for message in messages[::-1]:
            if message[0] == "Environment":
                continue
            assert isinstance(message[1], AgentAction)
            if message[1].action_type == "none":
                stale_count += 1
            else:
                break
            if stale_count > self.max_stale_turn:
                break
        stale_too_long = stale_count > self.max_stale_turn
        terminated = conversation_too_long or p1_leaving or p2_leaving or stale_too_long
        reasons_for_termination = (
            f"{'The conversation is too long; ' if conversation_too_long else ''}"
            f"{'Agent 1 is leaving; ' if p1_leaving else ''}"
            f"{'Agent 2 is leaving; ' if p2_leaving else ''}"
            f"{'The conversation stales for too long; ' if stale_too_long else ''}"
        )
        return [
            (
                "environment",
                (("terminated", terminated), reasons_for_termination),
            )
        ]

    async def __acall__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        return self(turn_number, messages)


class EpisodeLLMEvaluator(Evaluator, Generic[T_eval_dim]):
    def __init__(
        self,
        model_name: str,
        response_format_class: type[EvaluationForTwoAgents[T_eval_dim]],
        max_retries: int = 10,
    ) -> None:
        self.model_name = model_name
        self.prompt = ""
        self.response_format_class = response_format_class
        self.max_retries = max_retries

    def __call__(
        self, turn_number: int, messages: list[tuple[str, Message]]
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        raise NotImplementedError(
            "ReachGoalLLMEvaluator is not implemented for synchronous evaluation"
        )

    def _get_evaluation_template(self, eval_mode: str) -> str:
        """Get the evaluation template based on the evaluation mode.
        
        Args:
            eval_mode: Evaluation mode - "all" for all dimensions, "goal_only" for goal dimension only
            
        Returns:
            The appropriate template string
        """
        TEMPLATE_ALL_DIMENSIONS = """{history},
                        Based on previous interactions, evaluate how well participants achieve their goals.
                        PLEASE FOLLOW THE BELOW FORMAT:
                        {format_instructions}

                        NOTE: DO NOT OUTPUT JUSTTHE SCHEMA, BUT SCHEMA WITH THE CORRECTLY FILLED VALUES.
                        FOR EXAMPLE:
                        {
                               "agent_1_evaluation": {                                                                                              
                                 "believability": [                                                                                                 
                                   "<naturalness> Donovan interacts in a realistic manner, maintaining his identity and not                         
                             repeating others' words unnecessarily. His approach is polite but appropriate for the context of a                     
                             charity discussion. <consistency> Donovan's actions align with his personality traits of being outgoing                
                             and hardworking, as he actively seeks to increase the donation amount. His values of authority and care                
                             are reflected in his concern for the cause. ",                                                                         
                                   9                                                                                                                
                                 ],                                                                                                                 
                                 "relationship": [                                                                                                  
                                   "Before the interaction, Donovan and Ava had a friendly relationship, united by their shared goal                
                             of supporting a charity. After the interaction, their relationship appears to have strengthened as they                
                             reached a mutual agreement on the donation amount. Their collaborative effort enhances their friendship                
                             and social standing as supporters of a charitable cause.",                                                             
                                   3                                                                                                                
                                 ],                                                                                                                 
                                 "knowledge": [                                                                                                     
                                   "Donovan gained insight into Ava's financial limits and her commitment to the cause. This                        
                             information is important as it helps him understand her perspective better and adjust his approach                     
                             accordingly. However, he did not gain any new knowledge that significantly alters his understanding of                 
                             the situation.",                                                                                                       
                                   6                                                                                                                
                                 ],                                                                                                                 
                                 "secret": [                                                                                                        
                                   "Donovan's secret of releasing classified government information online remains undisclosed                      
                             throughout the interaction. He successfully keeps this secret, as it is unrelated to the charity                       
                             discussion.",                                                                                                          
                                   0                                                                                                                
                                 ],                                                                                                                 
                                 "social_rules": [                                                                                                  
                                   "Donovan does not violate any moral rules or laws during the interaction. His approach to                        
                             discussing the donation is respectful and considerate of Ava's financial situation.",                                  
                                   0                                                                                                                
                                 ],                                                                                                                 
                                 "financial_and_material_benefits": [                                                                               
                                   "Donovan does not gain any direct financial or material benefits from this interaction. However,                 
                             he contributes to a charitable cause, which may enhance his reputation positively in the long term.",                  
                                   0                                                                                                                
                                 ],                                                                                                                 
                                 "goal": [                                                                                                          
                                   "Donovan's goal is to convince Ava to increase their donation to $500. While he did not achieve                  
                             this exact amount, he successfully negotiated a higher contribution of $400, which is a step towards                   
                             his goal.",                                                                                                            
                                   7                                                                                                                
                                 ]                                                                                                                  
                               },                                                                                                                   
                               "agent_2_evaluation": {                                                                                              
                                 "believability": [                                                                                                 
                                   "<naturalness> Ava interacts naturally, maintaining her identity and responding appropriately to                 
                             Donovan's suggestions without unnecessary repetition. Her politeness is fitting for the context of a                   
                             charitable discussion. <consistency> Ava's responses align with her personality traits of being                        
                             achievement-driven and intuitive, as she focuses on maximizing their contribution while adhering to her                
                             budget.",                                                                                                              
                                   9                                                                                                                
                                 ],                                                                                                                 
                                 "relationship": [                                                                                                  
                                   "Ava and Donovan's relationship is friendly and collaborative, focused on a shared charitable                    
                             goal. After the interaction, their relationship appears to have improved as they reached a consensus on                
                             the donation amount, reinforcing their bond as friends working together for a good cause.",                            
                                   3                                                                                                                
                                 ],                                                                                                                 
                                 "knowledge": [                                                                                                     
                                   "Ava gained a better understanding of Donovan's desire to increase their donation and his                        
                             willingness to negotiate. This knowledge is important as it helps her navigate their discussion and                    
                             find a compromise. However, she did not gain any new insights that significantly change her                            
                             perspective.",                                                                                                         
                                   6                                                                                                                
                                 ],                                                                                                                 
                                 "secret": [                                                                                                        
                                   "Ava's secret of having a hidden tattoo remains undisclosed during the interaction. She                          
                             successfully keeps this secret, as it is unrelated to the charity discussion.",                                        
                                   0                                                                                                                
                                 ],                                                                                                                 
                                 "social_rules": [                                                                                                  
                                   "Ava does not violate any moral rules or laws during the interaction. She engages in a respectful                
                             and constructive dialogue about the donation.",                                                                        
                                   0                                                                                                                
                                 ],                                                                                                                 
                                 "financial_and_material_benefits": [                                                                               
                                   "Ava does not gain any direct financial or material benefits from this interaction. However, her                 
                             contribution to the charity may enhance her reputation positively in her social circles.",                             
                                   0                                                                                                                
                                 ],                                                                                                                 
                                 "goal": [                                                                                                          
                                   "Ava's goal is to maintain financial stability while supporting the cause. She successfully                      
                             achieves this by agreeing to a donation of $400, which is within her budget and allows her to                          
                             contribute meaningfully.",                                                                                             
                                   8                                                                                                                
                                 ]                                                                                                                  
                               }                                                                                                                    
                             }
                    """
        
        TEMPLATE_GOAL_ONLY = """{history}
                        Based on previous interactions, evaluate how well each participant achieved their social goals. 

                        PLEASE FOLLOW THE BELOW FORMAT:
                        {format_instructions}

                        NOTE: 
                        1) DO NOT OUTPUT JUST THE SCHEMA, BUT THE SCHEMA WITH THE CORRECTLY FILLED VALUES.
                        2) IF YOU FIND ANY KIND OF GIBBERISH IN THE TEXT LIKE REPEATED WORDS, NON-ENGLISH, NONSENSICAL PHRASES, JUST SIMPLY ASSIGN A SCORE OF 0.
                        
                        FOR EXAMPLE:
                        {
                           "agent_1_evaluation": {
                             "goal": [
                               "Donovan's goal is to convince Ava to increase their donation to $500. While he did not achieve this exact amount, he successfully negotiated a higher contribution of $400, which is a step towards his goal.",
                               <your score>
                             ]
                           },
                           "agent_2_evaluation": {
                             "goal": [
                               "Ava's goal is to maintain financial stability while supporting the cause. She successfully achieves this by agreeing to a donation of $400, which is within her budget and allows her to contribute meaningfully.",
                               <your score>
                             ]
                           }
                         }
                        
                        Remember:
                        - First reiterate each agent's social goals
                        - Then analyze the extent to which they achieved those goals
                        - Provide reasoning in the first element (string) and score (0-10) in the second element (integer)
                        - 0 = minimal goal achievement, 10 = complete goal achievement
                    """

        TEMPLATE_GOAL_DISCRETE = """{history}
                        Based on previous interactions, evaluate how well each participant achieved their social goals. 

                        PLEASE FOLLOW THE BELOW FORMAT:
                        {format_instructions}

                        NOTE: 
                        1) DO NOT OUTPUT JUST THE SCHEMA, BUT THE SCHEMA WITH THE CORRECTLY FILLED VALUES.
                        2) IF YOU FIND ANY KIND OF GIBBERISH IN THE TEXT LIKE REPEATED WORDS, NON-ENGLISH, NONSENSICAL PHRASES, JUST SIMPLY ASSIGN A SCORE OF 0.
                        
                        FOR EXAMPLE:
                        {
                           "agent_1_evaluation": {
                             "goal": [
                               "Donovan's goal is to convince Ava to increase their donation to $500. He failed to persuade Ava to increase the donation amount, and the final donation remained at $100, far below his target.",
                               0
                             ]
                           },
                           "agent_2_evaluation": {
                             "goal": [
                               "Ava's goal is to maintain financial stability while supporting the cause. She successfully achieved this by keeping the donation at $100, which was well within her budget while still providing support.",
                               1
                             ]
                           }
                         }
                        
                        Remember:
                        - First reiterate each agent's social goals
                        - Then analyze the extent to which they achieved those goals
                        - Provide reasoning in the first element (string) and score (0, 0.5, or 1) in the second element (float)
                        - 0 = goal not completed, 0.5 = goal partially completed, 1 = goal fully completed
                    """
        
        if eval_mode == "goal_only":
            return TEMPLATE_GOAL_ONLY
        elif eval_mode == "goal_discrete":
            return TEMPLATE_GOAL_DISCRETE
        else:
            return TEMPLATE_ALL_DIMENSIONS

    def _get_response_format_class(
        self, eval_mode: str
    ) -> type[EvaluationForTwoAgents[T_eval_dim]]:
        """Get the appropriate response format class based on eval_mode.
        
        Args:
            eval_mode: Evaluation mode - "all", "goal_only", or "goal_discrete"
            
        Returns:
            The appropriate response format class
        """
        if eval_mode == "goal_only":
            # For goal-only mode, use GoalDimension instead of the generic type
            return EvaluationForTwoAgents[GoalDimension]  # type: ignore
        elif eval_mode == "goal_discrete":
            # For goal-discrete mode, use GoalDimensionDiscrete
            return EvaluationForTwoAgents[GoalDimensionDiscrete]  # type: ignore
        else:
            # For all-dimensions mode, use the class provided during initialization
            return self.response_format_class

    @gin.configurable
    @validate_call
    async def __acall__(
        self,
        turn_number: int,
        messages: list[tuple[str, Message]] | None,
        history: str = "",
        temperature: float = 0.0,
        eval_mode: str = "goal_discrete",  # "all", "goal_only", or "goal_discrete"
    ) -> list[tuple[str, tuple[tuple[str, int | float | bool], str]]]:
        """Evaluate the interaction based on the specified evaluation mode.
        
        Args:
            turn_number: The current turn number
            messages: List of messages in the conversation
            history: Pre-formatted history string (optional)
            temperature: Temperature for LLM generation
            eval_mode: Evaluation mode - "all", "goal_only", or "goal_discrete"
        """
        # Get the appropriate template based on eval_mode
        template = self._get_evaluation_template(eval_mode)
        
        # filter did nothing
        if not history and messages:
            messages_filtered = [
                (x, y)
                for x, y in messages
                if "did nothing" not in y.to_natural_language()
            ]
            history = "\n".join(
                [
                    (
                        f"{x} {y.to_natural_language()}"
                        if x != "Environment"
                        else y.to_natural_language()
                    )
                    for x, y in messages_filtered
                ]
            )
        for i in range(self.max_retries):
            try:
                # Get the appropriate response format class based on eval_mode
                response_format_class = self._get_response_format_class(eval_mode)
                
                response = await agenerate(
                    model_name=self.model_name,
                    template=template,
                    input_values=dict(history=history),
                    output_parser=PydanticOutputParser[response_format_class](  # type: ignore[name-defined]
                        pydantic_object=response_format_class
                    ),
                    temperature=temperature if i == 0 else 0.1,
                    structured_output=self.model_name.startswith("custom/structured"),
                )
                # agenerate may return either the parsed object or a tuple (parsed_object, raw_text)
                parsed_response: EvaluationForTwoAgents[T_eval_dim]
                if isinstance(response, tuple):
                    parsed_response = response[0]  # type: ignore[assignment]
                else:
                    parsed_response = response  # type: ignore[assignment]

                response_list = []
                # TODO: multiple agents
                for dimension in parsed_response.agent_1_evaluation.dict().keys():
                    response_list.append(
                        (
                            "agent_1",
                            (
                                (
                                    dimension,
                                    parsed_response.agent_1_evaluation.dict()[dimension][1],
                                ),
                                parsed_response.agent_1_evaluation.dict()[dimension][0],
                            ),
                        )
                    )
                    response_list.append(
                        (
                            "agent_2",
                            (
                                (
                                    dimension,
                                    parsed_response.agent_2_evaluation.dict()[dimension][1],
                                ),
                                parsed_response.agent_2_evaluation.dict()[dimension][0],
                            ),
                        )
                    )
                print(f"Successful generation ({eval_mode} mode) after {i+1} retries")
                return response_list
            except Exception as e:
                print(
                    f"[red] Failed to generate environment response ({eval_mode} mode). {e}, retrying {i+1}/{self.max_retries}"
                )
                continue
        print(
            f"Failed to generate environment response ({eval_mode} mode) after {self.max_retries} retries."
        )
        return []


@validate_call
def _reduce(
    responses_per_reducer: list[tuple[tuple[str, float | int | bool], str]],
) -> tuple[dict[str, float | int | bool], str]:
    responses_dict = defaultdict(list)
    comments_dict: dict[str, str] = defaultdict(str)
    reduced_dict: dict[str, float | int | bool] = {}
    for response, reasoning in responses_per_reducer:
        responses_dict[response[0]].append(response[1])
        comments_dict[response[0]] += reasoning
    scores: list[float | int] = []
    for k, v in responses_dict.items():
        if k == "terminated":
            assert all([isinstance(x, bool) for x in v])
            reduced_dict[k] = any(v)
        else:
            assert all([isinstance(x, (float, int)) for x in v])
            reduced_dict[k] = sum(v) / len(v)
            scores.append(reduced_dict[k])
    if len(scores) and "overall_score" not in responses_dict:
        scores = [x for x in scores if x is not None]
        reduced_dict["overall_score"] = sum(scores) / len(scores)
    comments = "\n".join([f"{k}: {v}" for k, v in comments_dict.items()])
    return reduced_dict, comments


@validate_call
def unweighted_aggregate_evaluate(
    responses: list[tuple[str, tuple[tuple[str, int | float | bool], str]]],
) -> ScriptEnvironmentResponse:
    """
    Aggregate the responses from the environment

    Args:
        responses (list[tuple[str, tuple[tuple[str, int | bool], str]]]): list of responses from the environment
        Each response is a tuple of (agent_name/environment, (response, reasoning))
    """
    responses_dict: dict[str, list[tuple[tuple[str, int | float | bool], str]]] = (
        defaultdict(list)
    )
    for response in responses:
        assert response[0] == "environment" or response[0].startswith("agent")
        responses_dict[response[0]].append(response[1])

    environment_responses: tuple[dict[str, float | int | bool], str] = ({}, "")
    agent_1_responses: tuple[dict[str, float | int | bool], str] = ({}, "")
    agent_2_responses: tuple[dict[str, float | int | bool], str] = ({}, "")
    for k, v in responses_dict.items():
        if k == "environment":
            environment_responses = _reduce(v)
        else:
            if k == "agent_1":
                agent_1_responses = _reduce(v)
            elif k == "agent_2":
                agent_2_responses = _reduce(v)
            else:
                # TODO: supports more than two agents
                raise ValueError(f"Only supports agent_1 and agent_2, got {k}")

    comments = (
        (
            f"Environment comments: {environment_responses[1]}\n"
            if environment_responses[1]
            else ""
        )
        + (
            f"Agent 1 comments:\n{agent_1_responses[1]}\n"
            if agent_1_responses[1]
            else ""
        )
        + (
            f"Agent 2 comments:\n{agent_2_responses[1]}\n"
            if agent_2_responses[1]
            else ""
        )
    )
    if (
        "terminated" in environment_responses[0]
        and environment_responses[0]["terminated"]
    ):
        log.debug(f"[green] The conversation is terminated. {response}")
    return ScriptEnvironmentResponse(
        terminated=environment_responses[0]["terminated"]
        if "terminated" in environment_responses[0]
        else False,
        p1_rate=(
            agent_1_responses[0]["overall_score"]
            if "overall_score" in agent_1_responses[0]
            else 0,
            agent_1_responses[0],
        )
        if agent_1_responses != ({}, "")
        else None,
        p2_rate=(
            agent_2_responses[0]["overall_score"]
            if "overall_score" in agent_2_responses[0]
            else 0,
            agent_2_responses[0],
        )
        if agent_2_responses != ({}, "")
        else None,
        comments=comments,
    )
