import asyncio
from typing import List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from ..llms.embedding_model import EmbeddingModel
from ..llms.llm_model import LLMModel
from ..store.store import VectorStore
from ..utils.cache import Cache
from ..utils.logger import CostLogger
from ..utils.policies import VALID_POLICIES
from .gather_evidence import gather_evidence
from .retrieve_evidence import retrieve_evidence
from .retrieve_policy_plans_and_riders import retrieve_policy_plans_and_riders
from .retrieve_premiums import VALID_COVERAGE, VALID_PLANS, retrieve_premiums
from .utils import tool_metadata


class PaperQAToolSpec(BaseToolSpec):
    spec_functions = [
        "gather_evidence_by_query",
        "gather_policy_overview",
        "retrieve_evidence",
        "retrieve_policy_plans_and_riders",
        "retrieve_premiums",
    ]
    store: VectorStore
    cache: Cache
    embedding_model: EmbeddingModel
    summary_llm_model: LLMModel
    current_task_id: str = ""
    cost_logger: CostLogger

    def __init__(
        self,
        store: VectorStore,
        cache: Cache,
        embedding_model: EmbeddingModel,
        summary_llm_model: LLMModel,
        cost_logger: Optional[CostLogger] = None,
    ):
        super().__init__()
        self.store = store
        self.cache = cache
        self.embedding_model = embedding_model
        self.summary_llm_model = summary_llm_model
        self.cost_logger = cost_logger or CostLogger()

    @tool_metadata(
        desc=f"""
Retrieve list of plans and riders for specific policies.

Args:
    policies (List[str]) = None: A list of policies from {VALID_POLICIES}.
""",
        output_desc="Retrieving plans and riders for {policies}...",
    )
    def retrieve_policy_plans_and_riders(self, policies: List[str]) -> str:
        return retrieve_policy_plans_and_riders(
            policies,
        )

    @tool_metadata(
        desc=f"""
Find and return pieces of evidence that are relevant
to a given query and policy.

Always provide a policy where possible.

This can be called multiple times with varying search terms
if insufficient information was found.
This should only be called for queries relevant to insurance.
Never ever call this tool for queries unrelated to insurance.

valid_policies = {VALID_POLICIES}

Args:
    query (str): The query to search for
    policy (str) = None: The policy to filter by, must be one of values in valid_policies list. If None, defaults to searching all policies.
""",
        output_desc='Retrieving information with query "{query}" on policy "{policy}"...',
        default_kwargs={"policy": None},
    )
    def gather_evidence_by_query(
        self,
        query: str,
        policy: Optional[str] = None,
    ) -> str:
        async def gather_evidence_helper():
            response = await gather_evidence(
                self.cache,
                self.store,
                query=query,
                policy=policy,
                embedding_model=self.embedding_model,
                summary_llm_model=self.summary_llm_model,
                prefix=self.current_task_id,
            )
            return response

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(gather_evidence_helper())

    @tool_metadata(
        desc=f"""
Find all information about a policy.
This is useful when you want to get key features about a policy,
summarize a policy, or broadly compare different policies.

valid_policies = {VALID_POLICIES}

Args:
    policy (str) = None: The policy to filter by, must be one of values in valid_policies list. If None, defaults to searching all policies.
""",
        output_desc='Retrieving all information about policy "{policy}"...',
    )
    def gather_policy_overview(
        self,
        policy: str,
    ) -> str:
        async def gather_evidence_helper():
            response = await gather_evidence(
                self.cache,
                self.store,
                query=None,  # Use query=None to return all information
                policy=policy,
                embedding_model=self.embedding_model,
                summary_llm_model=self.summary_llm_model,
                prefix=self.current_task_id,
            )
            return response

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(gather_evidence_helper())

    @tool_metadata(
        output_desc="Retrieving gathered evidence...",
    )
    def retrieve_evidence(self, question: str) -> str:
        """
        Retrieves the evidence and summaries from earlier steps and
        combine them with the user's question to form an instruction
        to generate the final response.

        This should be called after all evidence has been gathered.

        Args:
            question (str): the question to be answered

        """
        return retrieve_evidence(
            self.cache,
            self.store,
            question,
        )

    @tool_metadata(
        desc=f"""
Retrieve insurance premiums.
Always use this tool when the user asks about the amount of premiums to be paid.
When this tool is used, always reply the user with the table returned by this tool without modification.

This tool will show the premiums payable, divided into two components.
- MediShield Life Premiums (which are fully payable by Medisave)
- Additional Insurance Premium (which is the additional premium payable by the user)

Insurance premiums depend on age, policy, and plan or coverage selected.

The coverage can be one of:
- Basic: the basic MediShield Life coverage that aims to cover Class B2 and Class C wards in public hospitals
- Standard Class B1: a standard policy that aims to cover up to Class B1 wards in public hospitals and has benefits set by the government
- Class B1: similar to Standard Class B1 (covers up to Class B1 wards in public hospitals) but has more benefits
- Class A: similar to Class B1 but covers up to Class A wards in public hospitals and has more benefits
- Private: similar to Class A but also covers private hospitals and has more benefits

Args:
    ages (List[int]) = None: The ages to retrieve premiums. If None, defaults to [10, 30, 50, 70].
    policies (List[str]) = None: A list of up to 3 policies from {VALID_POLICIES} or None. If None, defaults to all.
    plans (List[str]) = None: A list of plans from {VALID_PLANS} or None. If None, defaults to all.
    coverages (List[str]) = None: A list of coverages from {VALID_COVERAGE} or None. Ignored if plans are provided. If None, defaults to all.
""",
        output_desc="Retrieving premiums based on filters age={ages}, policies={policies}, plans={plans}, coverages={coverages}...",
        default_kwargs={
            "ages": None,
            "policies": None,
            "plans": None,
            "coverages": None,
        },
    )
    def retrieve_premiums(
        self,
        ages: Optional[List[int]] = None,
        policies: Optional[List[str]] = None,
        plans: Optional[List[str]] = None,
        coverages: Optional[List[str]] = None,
    ):
        plans = plans or []
        if "MediShield Life" in policies:
            plans.append("MediShield Life")
            policies.remove("MediShield Life")
        # Map policies to companies
        policies_map = {
            "NTUC Income IncomeShield Standard": "Income",
            "NTUC Income Enhanced IncomeShield": "Income",
            "NTUC Income IncomeShield": "Income",
            "AIA HealthShield Gold Max": "AIA",
            "Great Eastern GREAT SupremeHealth": "Great Eastern",
            "HSBC Life Shield": "HSBC",
            "Prudential PRUShield": "Prudential",
            "Raffles Shield": "Raffles",
            "Singlife Shield": "Singlife",
        }
        policies = policies or []
        if len(policies) > 3:
            return "Only enter up to 3 policies for the retrieve_premiums tool"
        companies = list(set(policies_map.get(p) for p in policies))
        # Default ages
        ages = ages or [10, 30, 50, 70]
        data = retrieve_premiums(ages, companies, plans, coverages, format="table")
        return f"""
{data}

In your response to the user, always include this table in your response word for word without modification.
"""


# Important: Note that any "Class" in the plan name may not correspond to the ward coverage "Class".
# For example, a Class A plan might only provide coverage for Class B1 wards.
