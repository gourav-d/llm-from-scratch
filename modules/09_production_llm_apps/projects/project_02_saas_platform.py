# =============================================================================
# Project 02: Multi-Tenant SaaS Platform
# Module 09 - Production LLM Applications
# =============================================================================
# GOAL:
#   Build a multi-tenant LLM SaaS platform simulation.
#   Multiple companies ("tenants") each have isolated data, quotas, and billing.
#   This is how products like Azure OpenAI Service, Salesforce Einstein, and
#   corporate ChatGPT deployments work at their core.
#
# WHY MULTI-TENANCY MATTERS:
#   A SaaS product serves MANY customers from ONE codebase and ONE deployment.
#   The challenge: customers must NEVER see each other's data, usage, or bills.
#   Multi-tenancy solves this with isolation at every level.
#
# C# ANALOGY:
#   Think of this as an ASP.NET Core app where:
#     - TenantRegistry  = a SQL table of customer accounts
#     - TenantContext   = a scoped DI container per request (IServiceScope)
#     - BillingEngine   = Azure metered billing / consumption-based pricing
#     - TenantRouter    = a middleware that reads a tenant header and resolves context
#     - PlatformAPI     = the top-level controller that orchestrates it all
#
# =============================================================================
# GLOSSARY
# =============================================================================
#
#  MULTI-TENANCY
#    A software architecture where a SINGLE application instance serves
#    MULTIPLE customers (tenants), each believing they have their own system.
#    C# analogy: One ASP.NET Core app serving Acme Corp AND TechStart from
#                the same server, but with separate database schemas.
#
#  TENANT
#    A single customer organization using the platform.
#    "Tenant" = the paying company, NOT individual users within that company.
#    C# analogy: One row in a Tenants database table.
#
#  TENANT ISOLATION
#    The guarantee that tenant A's data cannot be read, modified, or seen
#    by tenant B, even if they share the same infrastructure.
#    C# analogy: Row-Level Security (RLS) in SQL Server, or separate
#                database schemas (one schema per tenant).
#
#  DATA PARTITION
#    Dividing data so each tenant's records are separated from others.
#    Methods: separate databases, separate schemas, or a TenantId column.
#    C# analogy: WHERE TenantId = @currentTenantId in every LINQ query.
#
#  SUBSCRIPTION TIER
#    A plan level (e.g., Starter, Professional, Enterprise) that determines
#    what features and limits a tenant gets.
#    C# analogy: An enum SubscriptionTier { Starter, Professional, Enterprise }
#                stored on the Tenant entity.
#
#  BILLING
#    Charging tenants for their usage of the platform.
#    Two components: (1) flat monthly subscription + (2) usage-based metering.
#    C# analogy: Stripe integration or Azure metered billing.
#
#  METERING
#    Precisely measuring and recording resource consumption (tokens, API calls,
#    storage) so it can be billed accurately.
#    C# analogy: Incrementing a usage counter in Azure Metered Billing SDK.
#
#  USAGE REPORT
#    A per-tenant summary of how many tokens were used, how many requests
#    were made, and what the total cost is for the billing period.
#    C# analogy: A SSRS report or Power BI dashboard query over usage tables.
#
#  TENANT ADMIN
#    A user within a tenant who has elevated permissions within their own
#    tenant (can add users, see invoices) but cannot see other tenants.
#    C# analogy: A role "TenantAdmin" scoped to one tenant's data.
#
#  RESOURCE QUOTA
#    The maximum amount of a resource (tokens, users, API calls) a tenant
#    can use per billing period based on their subscription tier.
#    C# analogy: A plan limit check before every service operation.
#
#  COST ALLOCATION
#    Tracking which tenant caused which costs, so billing is accurate.
#    Critical in shared infrastructure (one GPU serving many tenants).
#    C# analogy: Azure Cost Management tags + resource group per tenant.
#
# =============================================================================
# ARCHITECTURE DIAGRAM
# =============================================================================
#
#   [Tenant A requests] ---> [Tenant Router] ---> [Tenant A context]  \
#                                    |                                  \
#   [Tenant B requests] ---> [Tenant Router] ---> [Tenant B context] ---+---> [Billing Engine]
#                                    |                                  /          |
#   [Tenant C requests] ---> [Tenant Router] ---> [Tenant C context]  /           v
#                                    |                                      [Usage Reports]
#                                    |
#                            [Tenant isolation:
#                             A cannot see B,
#                             B cannot see C,
#                             C cannot see A]
#
# How the Tenant Router works:
#
#   api_key in request
#       |
#       v
#   Look up key in _key_to_tenant index
#       |
#       +-- Found --> Get TenantContext for that tenant
#       |                  |
#       |                  v
#       |             Check tenant is active (not suspended)
#       |                  |
#       |                  v
#       |             Return isolated TenantContext
#       |
#       +-- Not Found --> Return None (401 Unauthorized)
#
# =============================================================================

import time          # For timestamps and simulated latency
import uuid          # For generating unique tenant and user IDs
import random        # For simulated response variation
import datetime      # For billing period calculations
from collections import defaultdict  # Auto-initialized dict (like Dictionary<K,V> with default)

# =============================================================================
# PART 1 -- SUBSCRIPTION TIERS
# =============================================================================
# Defines the three pricing tiers available to tenants.
# Each tier has different limits and costs.
#
# C# analogy: A SubscriptionTierConfig class with static readonly instances,
#             or values loaded from appsettings.json per tier.
# =============================================================================

# Each tier is a plain dict with all the limits for that tier.
# C# analogy: Dictionary<string, SubscriptionConfig> Tiers = new() { ... }
SUBSCRIPTION_TIERS = {

    "starter": {
        "name"              : "Starter",
        "price_monthly_usd" : 29,          # Flat monthly fee in USD
        "token_limit_month" : 100_000,     # 100K tokens per month (0 = unlimited)
        "max_users"         : 5,           # Max users in this tenant
        "rate_limit_per_min": 10,          # API requests per minute
        "support_level"     : "email",     # Type of support included
    },

    "professional": {
        "name"              : "Professional",
        "price_monthly_usd" : 99,
        "token_limit_month" : 1_000_000,   # 1M tokens per month
        "max_users"         : 25,
        "rate_limit_per_min": 60,
        "support_level"     : "chat",
    },

    "enterprise": {
        "name"              : "Enterprise",
        "price_monthly_usd" : 499,
        "token_limit_month" : 0,           # 0 = unlimited
        "max_users"         : 0,           # 0 = unlimited
        "rate_limit_per_min": 300,
        "support_level"     : "dedicated", # Dedicated account manager
    },
}

# Per-thousand-tokens cost for usage-based billing.
# Enterprise pays less per token because they pay more upfront.
# C# analogy: decimal GetCostPerThousandTokens(SubscriptionTier tier)
COST_PER_1K_TOKENS = {
    "starter"     : 0.005,    # $0.005 per 1,000 tokens
    "professional": 0.003,    # $0.003 per 1,000 tokens
    "enterprise"  : 0.002,    # $0.002 per 1,000 tokens
}


# =============================================================================
# PART 2 -- TENANT REGISTRY
# =============================================================================
# Stores all registered tenants and provides lookup methods.
# In production this is a database table with a TenantId primary key.
#
# C# analogy: ITenantRepository backed by Entity Framework Core,
#             with a Tenants DbSet<Tenant>.
# =============================================================================

class TenantRegistry:
    """
    Manages all registered tenants on the platform.
    Provides tenant creation, lookup, and API key management.
    C# analogy: A TenantService class with IQueryable<Tenant> from EF Core.
    """

    def __init__(self):
        """
        Initialize the registry with empty storage.
        C# analogy: private Dictionary<string, Tenant> _tenants = new();
        """
        self._tenants = {}          # tenant_id -> tenant dict
        self._key_index = {}        # api_key -> tenant_id (reverse lookup)
        self._seed_tenants()        # Add the pre-built demo tenants

    def _seed_tenants(self):
        """
        Pre-registers 4 demo tenants so the simulation has data to work with.
        C# analogy: DbContext.Database.EnsureCreated() + seed data in OnModelCreating().
        """

        # Register 4 companies on different tiers
        # C# analogy: _context.Tenants.AddRange(seedData); _context.SaveChanges();
        self.register_tenant("Acme Corp",    "enterprise")    # Big company, unlimited
        self.register_tenant("TechStart Inc","starter")       # Small startup, 100K tokens
        self.register_tenant("MegaCorp",     "enterprise")    # Another big company
        self.register_tenant("DataFlow Ltd", "professional")  # Mid-size, 1M tokens

    def register_tenant(self, name: str, tier: str):
        """
        Creates a new tenant account on the platform.

        Parameters:
          name -- company name (e.g., "Acme Corp")
          tier -- subscription tier: "starter", "professional", "enterprise"

        Returns: tenant_id (a UUID string)

        C# analogy: async Task<string> CreateTenantAsync(CreateTenantRequest req)
        """

        # Generate a unique ID for this tenant
        tenant_id = "tenant_" + str(uuid.uuid4())[:8]  # e.g., "tenant_a3f8b2c1"

        # Generate an API key for this tenant
        # In production, use a cryptographically secure random generator
        api_key = f"sk_{name.replace(' ', '').lower()[:8]}_{str(uuid.uuid4())[:6]}"

        # Build the tenant record
        tenant = {
            "tenant_id" : tenant_id,
            "name"      : name,
            "tier"      : tier,                     # Which subscription plan
            "api_key"   : api_key,                  # The key clients will use
            "status"    : "active",                 # active / suspended / cancelled
            "created_at": datetime.datetime.utcnow().isoformat(),
            "users"     : [],                       # List of user email strings
        }

        # Store the tenant by its ID
        self._tenants[tenant_id] = tenant

        # Index the API key so we can look up the tenant from just the key
        self._key_index[api_key] = tenant_id

        return tenant_id  # Return ID so caller can reference this tenant

    def get_tenant(self, tenant_id: str):
        """
        Returns a tenant dict by its ID, or None if not found.
        C# analogy: await _context.Tenants.FindAsync(tenantId)
        """
        return self._tenants.get(tenant_id)

    def resolve_api_key(self, api_key: str):
        """
        Looks up which tenant owns a given API key.
        Returns tenant_id or None if key is not recognized.
        C# analogy: _context.Tenants.FirstOrDefaultAsync(t => t.ApiKey == apiKey)
        """
        return self._key_index.get(api_key)

    def list_tenants(self):
        """
        Returns a list of all registered tenant dicts.
        C# analogy: _context.Tenants.ToListAsync()
        """
        return list(self._tenants.values())  # Return copies in a list


# =============================================================================
# PART 3 -- TENANT CONTEXT (isolated data per tenant)
# =============================================================================
# Each TenantContext holds ONE tenant's private data.
# No other tenant can read or write to it.
#
# This is the core of tenant isolation:
#   - Tenant A gets TenantContext("tenant_a_id")
#   - Tenant B gets TenantContext("tenant_b_id")
#   - They are completely separate objects with no shared state
#
# C# analogy: A scoped DI service (AddScoped<ITenantContext>()) where the
#             scope is per-HTTP-request and the tenant is set from the JWT.
#             Or, a separate database schema per tenant.
# =============================================================================

class TenantContext:
    """
    Holds isolated data for one tenant: users, usage history, messages.
    Only code that has a reference to THIS tenant's context can see its data.
    C# analogy: An IServiceScope with a TenantId set in the scope's container,
                preventing any other scope from accessing this tenant's DbContext.
    """

    def __init__(self, tenant_id: str, tenant_name: str, tier: str):
        """
        Initialize an empty context for one tenant.
        C# analogy: Constructor injection of TenantId into a scoped service.
        """
        self._tenant_id   = tenant_id    # The owning tenant's ID (private)
        self._tenant_name = tenant_name  # Display name for reports
        self._tier        = tier         # Subscription tier

        # Users registered within this tenant
        # C# analogy: List<TenantUser> Users { get; private set; }
        self._users = []

        # Usage records: list of {date, tokens, cost, request_id}
        # C# analogy: List<UsageRecord> UsageHistory { get; }
        self._usage_records = []

        # Aggregate totals (pre-computed for fast reporting)
        self._total_tokens   = 0   # Running sum of all tokens used
        self._total_cost_usd = 0.0 # Running sum of all usage costs
        self._total_requests = 0   # Running count of all requests

    def add_user(self, email: str, role: str = "user"):
        """
        Adds a user to this tenant's account.
        Users belong to exactly ONE tenant.
        C# analogy: _context.TenantUsers.Add(new TenantUser { ... });

        Parameters:
          email -- user's email address
          role  -- "admin" or "user" within this tenant
        """
        # Check for duplicates (don't add the same email twice)
        existing_emails = [u["email"] for u in self._users]
        if email in existing_emails:
            return  # Already registered, skip

        # Add the user record
        self._users.append({
            "email"     : email,
            "role"      : role,
            "added_at"  : datetime.datetime.utcnow().isoformat(),
            "tenant_id" : self._tenant_id,   # Bind user to this tenant
        })

    def get_users(self):
        """
        Returns the list of users in this tenant.
        ISOLATION GUARANTEE: This only returns THIS tenant's users.
        C# analogy: _context.TenantUsers.Where(u => u.TenantId == _tenantId).ToList()
        """
        return list(self._users)  # Return a copy, not the internal list

    def record_usage(self, tokens: int, cost_usd: float, request_id: str = None):
        """
        Records usage for billing and reporting purposes.
        Called after every successful LLM request.
        C# analogy: await _usageRepository.AddAsync(new UsageRecord { ... });

        Parameters:
          tokens     -- number of tokens consumed
          cost_usd   -- dollar cost of this request
          request_id -- unique ID for this request (for audit trails)
        """
        # Build a usage record with full context
        record = {
            "date"      : datetime.date.today().isoformat(),  # "2026-04-30"
            "tokens"    : tokens,
            "cost_usd"  : cost_usd,
            "request_id": request_id or str(uuid.uuid4())[:8],
            "tenant_id" : self._tenant_id,   # Redundant but useful for auditing
        }

        # Store the record
        self._usage_records.append(record)

        # Update running totals for fast summary queries
        self._total_tokens   += tokens
        self._total_cost_usd += cost_usd
        self._total_requests += 1

    def get_usage_report(self):
        """
        Returns a summary of this tenant's usage.
        ISOLATION: Only shows this tenant's data, never another tenant's.
        C# analogy: _context.UsageRecords
                        .Where(r => r.TenantId == _tenantId)
                        .GroupBy(r => r.Date)
                        .Select(g => new DailyUsage { ... })
                        .ToListAsync()
        """

        # Build a "by_day" breakdown: date -> {tokens, cost, requests}
        by_day = {}
        for record in self._usage_records:
            date = record["date"]  # "2026-04-30"
            if date not in by_day:
                # First record for this day -- initialize the bucket
                by_day[date] = {"tokens": 0, "cost_usd": 0.0, "requests": 0}

            # Add this record's values to the day's totals
            by_day[date]["tokens"]   += record["tokens"]
            by_day[date]["cost_usd"] += record["cost_usd"]
            by_day[date]["requests"] += 1

        return {
            "tenant_id"       : self._tenant_id,
            "tenant_name"     : self._tenant_name,
            "tier"            : self._tier,
            "total_tokens"    : self._total_tokens,
            "total_cost_usd"  : round(self._total_cost_usd, 4),  # Round to 4 decimal places
            "total_requests"  : self._total_requests,
            "by_day"          : by_day,
        }


# =============================================================================
# PART 4 -- BILLING ENGINE
# =============================================================================
# Calculates costs, checks quotas, and generates invoices.
# In production this would integrate with Stripe or Azure Billing.
#
# C# analogy: A BillingService that calls Stripe SDK or Azure Cost Management,
#             with metered billing for usage-based charges.
# =============================================================================

class BillingEngine:
    """
    Handles all billing calculations for the platform.
    Calculates usage costs, checks quotas, and generates invoices.
    C# analogy: IBillingService backed by Stripe or an in-house billing system.
    """

    def __init__(self, tenant_registry: TenantRegistry, tenant_contexts: dict):
        """
        Initialize with references to the registry and all tenant contexts.
        C# analogy: Constructor injection of ITenantRepository and IUsageRepository.

        Parameters:
          tenant_registry  -- the TenantRegistry to look up tier info
          tenant_contexts  -- dict of tenant_id -> TenantContext
        """
        self._registry = tenant_registry    # To look up tier/quota info
        self._contexts = tenant_contexts    # To look up usage data

    def calculate_cost(self, tokens: int, tier: str):
        """
        Calculates the usage cost for a given number of tokens on a given tier.
        Formula: (tokens / 1000) * cost_per_1k_tokens

        Parameters:
          tokens -- how many tokens were used
          tier   -- "starter", "professional", or "enterprise"

        Returns: float (USD cost)

        C# analogy: decimal CalculateCost(int tokens, SubscriptionTier tier)
        """
        # Get the per-1K rate for this tier
        rate = COST_PER_1K_TOKENS.get(tier, 0.005)  # Default to starter rate if unknown

        # Calculate: (tokens / 1000) * rate
        # Example: 5,000 tokens on starter = (5000/1000) * 0.005 = $0.025
        cost = (tokens / 1000.0) * rate

        return round(cost, 6)  # Round to 6 decimal places for precision

    def check_quota(self, tenant_id: str):
        """
        Checks if a tenant is within their monthly token quota.
        Enterprise tenants have unlimited quota (token_limit_month == 0).

        Parameters:
          tenant_id -- which tenant to check

        Returns: dict with quota status details
        C# analogy: async Task<QuotaStatus> CheckQuotaAsync(string tenantId)
        """

        # Look up the tenant record
        tenant = self._registry.get_tenant(tenant_id)
        if tenant is None:
            return {"within_quota": False, "reason": "Tenant not found"}

        # Look up the tier configuration
        tier_config = SUBSCRIPTION_TIERS.get(tenant["tier"])
        if tier_config is None:
            return {"within_quota": False, "reason": "Unknown tier"}

        # Get the token limit for this tier
        token_limit = tier_config["token_limit_month"]

        # 0 = unlimited (Enterprise)
        if token_limit == 0:
            return {
                "within_quota": True,
                "tokens_used" : self._contexts[tenant_id]._total_tokens,
                "tokens_limit": "Unlimited",
                "pct_used"    : 0.0,
                "reason"      : "Unlimited quota (Enterprise)",
            }

        # Get how many tokens this tenant has actually used
        tokens_used = self._contexts[tenant_id]._total_tokens

        # Calculate percentage used
        pct_used = (tokens_used / token_limit) * 100 if token_limit > 0 else 0.0

        # Is the tenant within quota?
        within_quota = tokens_used < token_limit

        return {
            "within_quota": within_quota,
            "tokens_used" : tokens_used,
            "tokens_limit": token_limit,
            "pct_used"    : round(pct_used, 1),
            "reason"      : "OK" if within_quota else "Monthly token quota exceeded",
        }

    def generate_invoice(self, tenant_id: str):
        """
        Generates a billing invoice for a tenant's current period.
        Shows: subscription cost + usage cost = total.

        Parameters:
          tenant_id -- which tenant's invoice to generate

        Returns: invoice dict
        C# analogy: async Task<Invoice> GenerateInvoiceAsync(string tenantId)
        """

        # Look up the tenant
        tenant = self._registry.get_tenant(tenant_id)
        if tenant is None:
            return {"error": "Tenant not found"}

        # Get the tier and its config
        tier        = tenant["tier"]
        tier_config = SUBSCRIPTION_TIERS[tier]

        # Get the tenant's usage data
        context      = self._contexts[tenant_id]
        total_tokens = context._total_tokens
        total_usage_cost = context._total_cost_usd

        # Flat subscription fee
        subscription_cost = tier_config["price_monthly_usd"]

        # Total invoice = subscription + usage
        total = subscription_cost + total_usage_cost

        # Build the invoice
        invoice = {
            "tenant_id"        : tenant_id,
            "tenant_name"      : tenant["name"],
            "tier"             : tier_config["name"],
            "billing_period"   : datetime.date.today().strftime("%B %Y"),  # e.g., "April 2026"
            "subscription_cost": subscription_cost,
            "usage_cost"       : round(total_usage_cost, 4),
            "total_usd"        : round(total, 4),
            "line_items"       : [
                {
                    "description": f"{tier_config['name']} Plan (monthly)",
                    "amount_usd" : subscription_cost,
                },
                {
                    "description": f"Token usage: {total_tokens:,} tokens "
                                   f"@ ${COST_PER_1K_TOKENS[tier]}/1K",
                    "amount_usd" : round(total_usage_cost, 4),
                },
            ],
        }

        return invoice


# =============================================================================
# PART 5 -- TENANT ROUTER
# =============================================================================
# Routes incoming API requests to the correct tenant's isolated context.
# This is the "traffic cop" that ensures requests land in the right bucket.
#
# C# analogy: A middleware that reads the X-Tenant-ID header (or resolves it
#             from the API key), then sets ITenantContext in the DI scope.
# =============================================================================

class TenantRouter:
    """
    Routes API requests to the correct tenant context based on API key.
    The router is the ONLY place where a key gets mapped to a tenant.
    After routing, all code works only with the resolved TenantContext.
    C# analogy: TenantMiddleware that sets IHttpContextAccessor tenant data,
                or a multi-tenant resolver that configures EF Core's schema.
    """

    def __init__(self, tenant_registry: TenantRegistry, tenant_contexts: dict):
        """
        Initialize with the registry (to resolve keys) and contexts (to return).

        Parameters:
          tenant_registry  -- used to look up api_key -> tenant_id
          tenant_contexts  -- used to return the isolated TenantContext
        """
        self._registry = tenant_registry
        self._contexts = tenant_contexts

    def route(self, api_key: str):
        """
        Given an API key, returns the tenant's isolated context.

        Parameters:
          api_key -- the key from the incoming request

        Returns: (tenant_id: str, context: TenantContext) or (None, None)

        C# analogy: Task<(string tenantId, ITenantContext context)> ResolveAsync(string apiKey)
        """

        # Step 1: Look up which tenant owns this key
        tenant_id = self._registry.resolve_api_key(api_key)
        if tenant_id is None:
            return (None, None)  # Unknown key -- reject

        # Step 2: Look up the tenant record to check status
        tenant = self._registry.get_tenant(tenant_id)
        if tenant is None:
            return (None, None)  # Should not happen, but be safe

        # Step 3: Check the tenant is not suspended or cancelled
        if tenant["status"] != "active":
            return (None, None)  # Suspended tenant -- reject

        # Step 4: Return the tenant_id and their isolated context
        context = self._contexts.get(tenant_id)
        if context is None:
            return (None, None)  # No context found (should not happen)

        return (tenant_id, context)  # All good -- return isolated context


# =============================================================================
# PART 6 -- LLM SIMULATOR (shared across tenants, but isolated by context)
# =============================================================================
# In production, this would call a real LLM API.
# The same LLM is used for all tenants, but NEVER shares their conversation data.
# C# analogy: A shared IHttpClientFactory that makes isolated API calls,
#             never mixing one tenant's data into another's request.
# =============================================================================

# Canned responses for the simulated LLM
LLM_RESPONSES = {
    "help"    : "I can help you with API integration, data analysis, and automation.",
    "data"    : "Our platform processes your data with strict tenant isolation.",
    "billing" : "Your usage is billed per-token based on your subscription tier.",
    "api"     : "Use your tenant API key to authenticate all requests.",
    "error"   : "Check your API key and ensure you are within your quota limits.",
    "report"  : "Usage reports are available in real-time from the /reports endpoint.",
    "fallback": "Thanks for your message. How can I assist you today?",
}

def simulate_llm_call(message: str):
    """
    Simulates an LLM API call.
    Returns response text and token counts.

    C# analogy: async Task<LlmResponse> CallOpenAiAsync(string message)
    """

    # Simulate 50-200ms network latency
    time.sleep(random.uniform(0.05, 0.2))

    # Pick a response based on keywords in the message
    lowered = message.lower()
    response_text = LLM_RESPONSES["fallback"]   # Default to fallback

    for keyword in ["help", "data", "billing", "api", "error", "report"]:
        if keyword in lowered:
            response_text = LLM_RESPONSES[keyword]
            break  # Use the first matching keyword

    # Estimate tokens (1 token ~ 4 characters)
    input_tokens  = len(message) // 4
    output_tokens = len(response_text) // 4

    return {
        "response"     : response_text,
        "input_tokens" : input_tokens,
        "output_tokens": output_tokens,
        "total_tokens" : input_tokens + output_tokens,
    }


# =============================================================================
# PART 7 -- PLATFORM API
# =============================================================================
# Top-level entry point that brings everything together.
# Identifies the tenant, checks quotas, processes the request,
# and records billing -- all within the tenant's isolated context.
#
# C# analogy: A Controller action that uses ITenantContext (scoped service),
#             IBillingService, and an ILlmClient to handle the full flow.
# =============================================================================

class PlatformAPI:
    """
    The main SaaS Platform API.
    Routes requests to the correct tenant context, enforces quotas,
    calls the LLM, and records billing.
    C# analogy: ChatController with all dependencies injected.
    """

    def __init__(self):
        """
        Initialize all platform components.
        C# analogy: IServiceCollection setup in Program.cs.
        """

        # Create the tenant registry (the "database" of tenants)
        self._registry = TenantRegistry()

        # Create one TenantContext for each registered tenant
        # This is where isolation lives: one object per tenant, no sharing.
        self._contexts = {}   # tenant_id -> TenantContext
        for tenant in self._registry.list_tenants():
            tid = tenant["tenant_id"]
            self._contexts[tid] = TenantContext(
                tenant_id   = tid,
                tenant_name = tenant["name"],
                tier        = tenant["tier"],
            )

        # Create the router (resolves api_key -> tenant context)
        self._router = TenantRouter(self._registry, self._contexts)

        # Create the billing engine
        self._billing = BillingEngine(self._registry, self._contexts)

        # Counters for the overall platform (not per-tenant)
        self._platform_total_requests  = 0
        self._platform_quota_rejections = 0

    def handle_request(self, api_key: str, message: str):
        """
        Processes a single request from a tenant user.

        Steps:
          1. Resolve api_key to a tenant
          2. Check tenant quota
          3. Call LLM (simulated)
          4. Record usage for billing
          5. Return response

        Parameters:
          api_key -- the tenant's API key
          message -- the user's message

        Returns: response dict
        C# analogy: async Task<IActionResult> PostMessage([FromBody] MessageRequest req)
        """

        self._platform_total_requests += 1   # Count every attempt

        # -----------------------------------------------
        # Step 1: Route to tenant context
        # -----------------------------------------------
        tenant_id, context = self._router.route(api_key)

        if tenant_id is None:
            # Unknown or suspended tenant -- reject
            return {
                "success"    : False,
                "error"      : "Unauthorized: invalid or inactive tenant API key",
                "http_status": 401,
            }

        # -----------------------------------------------
        # Step 2: Check quota
        # -----------------------------------------------
        quota_status = self._billing.check_quota(tenant_id)

        if not quota_status["within_quota"]:
            # Quota exceeded -- reject without calling LLM
            self._platform_quota_rejections += 1
            return {
                "success"     : False,
                "tenant_id"   : tenant_id,
                "error"       : f"Quota exceeded: {quota_status['reason']}",
                "tokens_used" : quota_status["tokens_used"],
                "tokens_limit": quota_status["tokens_limit"],
                "http_status" : 402,   # 402 Payment Required = quota exceeded
            }

        # -----------------------------------------------
        # Step 3: Call the LLM (simulated)
        # -----------------------------------------------
        llm_result = simulate_llm_call(message)

        # -----------------------------------------------
        # Step 4: Record usage for billing
        # -----------------------------------------------
        # Calculate the cost for this request
        tenant  = self._registry.get_tenant(tenant_id)
        cost    = self._billing.calculate_cost(
            tokens = llm_result["total_tokens"],
            tier   = tenant["tier"],
        )

        # Record usage in the tenant's ISOLATED context
        request_id = str(uuid.uuid4())[:8]   # Short unique ID for this request
        context.record_usage(
            tokens     = llm_result["total_tokens"],
            cost_usd   = cost,
            request_id = request_id,
        )

        # -----------------------------------------------
        # Step 5: Return response
        # -----------------------------------------------
        return {
            "success"      : True,
            "tenant_id"    : tenant_id,
            "request_id"   : request_id,
            "response"     : llm_result["response"],
            "usage"        : {
                "input_tokens" : llm_result["input_tokens"],
                "output_tokens": llm_result["output_tokens"],
                "total_tokens" : llm_result["total_tokens"],
            },
            "cost_usd"     : cost,
            "http_status"  : 200,
        }


# =============================================================================
# DEMO SIMULATION
# =============================================================================

def run_demo():
    """
    Runs the full multi-tenant SaaS platform demonstration.
    Covers: tenant setup, user registration, requests, quota limits,
    billing reports, and isolation verification.
    C# analogy: Integration test class with [TestInitialize] and [TestMethod].
    """

    print("=" * 65)
    print("  Multi-Tenant SaaS LLM Platform - Demo Simulation")
    print("=" * 65)

    # -----------------------------------------------
    # PART A: Initialize the platform
    # -----------------------------------------------
    print("\n[1] Initializing platform (4 tenants pre-registered)...")
    api = PlatformAPI()   # All tenants created in TenantRegistry._seed_tenants()

    # Print the registered tenants for visibility
    print("\n  Registered Tenants:")
    for tenant in api._registry.list_tenants():
        tier_config = SUBSCRIPTION_TIERS[tenant["tier"]]
        print(f"    - {tenant['name']:<20} Tier: {tenant['tier']:<14} "
              f"Key: {tenant['api_key']}")

    # -----------------------------------------------
    # PART B: Add 5 users to each tenant
    # -----------------------------------------------
    print("\n[2] Adding 5 users to each tenant...")

    # Map names to tenants for easy lookup in the demo
    tenant_list = api._registry.list_tenants()

    # Build a name -> tenant dict for easy access
    # C# analogy: var tenantDict = tenants.ToDictionary(t => t.Name);
    tenants_by_name = {t["name"]: t for t in tenant_list}

    # Add 5 sample users to each tenant's context
    for tenant in tenant_list:
        tid     = tenant["tenant_id"]
        context = api._contexts[tid]
        name    = tenant["name"].replace(" ", "").lower()[:6]   # Abbreviated name

        for i in range(1, 6):   # Users 1 through 5
            email = f"user{i}@{name}.example.com"
            role  = "admin" if i == 1 else "user"   # First user is tenant admin
            context.add_user(email, role)

        user_count = len(context.get_users())
        print(f"    - {tenant['name']}: {user_count} users added")

    # -----------------------------------------------
    # PART C: Simulate 30 requests across all tenants
    # -----------------------------------------------
    print("\n[3] Simulating 30 requests across all tenants...")
    print("    (TechStart is on Starter tier: 100K token limit)")

    # Messages to simulate varied usage
    sample_messages = [
        "Can you help me with data analysis?",
        "How does your API work?",
        "What are the billing details?",
        "Show me an error report.",
        "Help me with automation tasks.",
        "Explain how data isolation works.",
    ]

    # We will track per-tenant request counts for the summary
    request_counts = defaultdict(int)   # tenant_name -> count
    quota_hits     = defaultdict(int)   # tenant_name -> quota rejections

    # Get API keys for each tenant
    acme_key     = tenants_by_name["Acme Corp"]["api_key"]
    techstart_key= tenants_by_name["TechStart Inc"]["api_key"]
    megacorp_key = tenants_by_name["MegaCorp"]["api_key"]
    dataflow_key = tenants_by_name["DataFlow Ltd"]["api_key"]

    # Interleave requests from all 4 tenants (simulates real concurrent traffic)
    # 8 requests each = 32 total; we stop at 30 by using a counter
    all_requests = (
        [("Acme Corp",     acme_key)     ] * 12 +   # Acme gets most traffic
        [("TechStart Inc", techstart_key)] * 10 +   # TechStart will hit quota
        [("MegaCorp",      megacorp_key) ] * 5  +
        [("DataFlow Ltd",  dataflow_key) ] * 3
    )
    all_requests = all_requests[:30]   # Cap at 30

    # Pre-seed TechStart's token usage to simulate a tenant that has nearly
    # exhausted its monthly quota during the current billing period.
    # In a real system these tokens would have accumulated from previous requests.
    # We start them at 99,900 tokens (out of 100,000 limit).
    #
    # How the quota check triggers:
    #   tokens_used (99,900) + estimated_tokens (200 per request) = 100,100
    #   100,100 > 100,000 limit --> request BLOCKED with HTTP 402
    #
    # Result: first 5 TechStart requests sneak through (they accumulate real tokens
    # from 99,900 up to ~100,007), then the estimate-check trips for requests 6-10.
    #
    # C# analogy: SELECT SUM(Tokens) FROM UsageRecords WHERE TenantId = @id
    #             AND Month = @currentMonth  --> compare to plan limit.
    techstart_id = tenants_by_name["TechStart Inc"]["tenant_id"]
    api._contexts[techstart_id]._total_tokens = 99_900   # Simulate near-limit usage

    print("\n    Req# | Tenant           | Status     | Tokens | Cost")
    print("    " + "-" * 55)

    for i, (tenant_name, api_key) in enumerate(all_requests, start=1):
        # Pick a message round-robin from the sample list
        message = sample_messages[i % len(sample_messages)]

        # Make the request
        result = api.handle_request(api_key, message)

        request_counts[tenant_name] += 1   # Count this request

        # Determine status label for printing
        if result["success"]:
            status = "OK"
            tokens = result["usage"]["total_tokens"]
            cost   = f"${result['cost_usd']:.4f}"
        else:
            status = result.get("http_status", "ERR")
            tokens = 0
            cost   = "-"
            if result.get("http_status") == 402:
                quota_hits[tenant_name] += 1

        print(f"    {i:>3}  | {tenant_name:<17}| {str(status):<10} | "
              f"{tokens:>6} | {cost}")

    # -----------------------------------------------
    # PART D: Per-tenant usage reports
    # -----------------------------------------------
    print("\n\n[4] Usage Reports Per Tenant")
    print("=" * 65)

    for tenant in api._registry.list_tenants():
        tid         = tenant["tenant_id"]
        context     = api._contexts[tid]
        report      = context.get_usage_report()
        quota_status= api._billing.check_quota(tid)
        invoice     = api._billing.generate_invoice(tid)
        tier_config = SUBSCRIPTION_TIERS[tenant["tier"]]
        hits        = quota_hits.get(tenant["name"], 0)
        req_total   = request_counts.get(tenant["name"], 0)
        succeeded   = req_total - hits

        print(f"\n  Tenant: {tenant['name']} ({tier_config['name'].upper()})")
        print(f"    Requests:    {req_total} total ({succeeded} succeeded, {hits} quota exceeded)")

        if quota_status["tokens_limit"] == "Unlimited":
            token_display = f"{report['total_tokens']:,} / Unlimited"
        else:
            token_display = (f"{report['total_tokens']:,} / "
                             f"{quota_status['tokens_limit']:,} "
                             f"({quota_status['pct_used']}% used)")

        print(f"    Tokens:      {token_display}")
        print(f"    Usage cost:  ${report['total_cost_usd']:.4f}")
        print(f"    Subscription:${invoice['subscription_cost']}/month")
        print(f"    Invoice:     ${invoice['total_usd']:.4f} total")

    # -----------------------------------------------
    # PART E: ISOLATION PROOF
    # -----------------------------------------------
    print("\n\n[5] Tenant Isolation Verification")
    print("=" * 65)

    # Get tenant IDs for Acme and TechStart
    acme_id     = tenants_by_name["Acme Corp"]["tenant_id"]
    techstart_id= tenants_by_name["TechStart Inc"]["tenant_id"]

    # Get Acme's context -- this ONLY has Acme's users
    acme_context     = api._contexts[acme_id]
    techstart_context= api._contexts[techstart_id]

    # Get the users each context reports
    acme_users     = acme_context.get_users()
    techstart_users= techstart_context.get_users()

    print(f"\n  Acme Corp users visible to Acme context:      {len(acme_users)}")
    for u in acme_users:
        print(f"    - {u['email']}")

    print(f"\n  TechStart users visible to TechStart context: {len(techstart_users)}")
    for u in techstart_users:
        print(f"    - {u['email']}")

    # PROOF: Try to access TechStart's data from Acme's context
    print("\n  Testing isolation: Can Acme's context see TechStart's users?")

    # Acme's context has no knowledge of TechStart's users.
    # Their user list is completely separate objects.
    techstart_emails_in_acme = [
        u["email"] for u in acme_users
        if "techstart" in u["email"] or "techstartinc" in u["email"]
    ]

    if len(techstart_emails_in_acme) == 0:
        print("  ISOLATION VERIFIED: Acme's context contains 0 TechStart users.")
        print("  Tenant isolation confirmed: Tenant A cannot see Tenant B's data.")
    else:
        print("  ISOLATION FAILURE: This should never happen!")

    # Also demonstrate that the usage totals are separate
    print(f"\n  Acme usage total:      {acme_context._total_tokens:,} tokens")
    print(f"  TechStart usage total: {techstart_context._total_tokens:,} tokens")
    print("  These are stored in separate TenantContext objects -- no sharing.")

    # -----------------------------------------------
    # PART F: KEY TAKEAWAYS
    # -----------------------------------------------
    print_key_takeaways()


# =============================================================================
# KEY TAKEAWAYS
# =============================================================================

def print_key_takeaways():
    """Prints 5 production lessons about multi-tenant SaaS architecture."""

    print("\n\n" + "=" * 65)
    print("  KEY TAKEAWAYS: Multi-Tenant SaaS")
    print("=" * 65)

    takeaways = [
        (
            "1. Tenant isolation is the #1 design priority in SaaS.",
            "   Every data access must be scoped to one tenant.",
            "   One leak (showing Tenant A's data to Tenant B) is a",
            "   catastrophic trust violation that can end your business.",
            "   C#: Use Row-Level Security in SQL or separate EF Core",
            "   DbContexts per tenant, never shared."
        ),
        (
            "2. Subscription tiers + metered billing is the standard model.",
            "   Tenants pay a flat monthly fee (Starter/Pro/Enterprise)",
            "   PLUS a per-token usage charge.",
            "   This lets you serve small and large customers from the",
            "   same infrastructure while recovering GPU costs.",
            "   C#: Stripe's metered billing or Azure Metered Billing SDK."
        ),
        (
            "3. Quota enforcement must happen BEFORE calling the LLM.",
            "   Calling the model costs money even if you reject the result.",
            "   Always check quota first, reject early, save compute.",
            "   C#: Check quota in a middleware before reaching the controller."
        ),
        (
            "4. The Tenant Router is the single source of truth for identity.",
            "   Map api_key -> tenant_id in ONE place only.",
            "   All downstream code works only with the resolved TenantContext.",
            "   Never pass raw api_keys deeper than the router.",
            "   C#: Set ITenantContext in middleware; controllers never see the key."
        ),
        (
            "5. Usage reports must be real-time for customer trust.",
            "   Customers need to see their consumption NOW, not next month.",
            "   Aggregate incrementally (running totals) rather than",
            "   scanning all records each time -- much faster at scale.",
            "   C#: CQRS pattern: writes go to EventStore; reads come from",
            "   pre-aggregated ReadModel tables updated by event handlers."
        ),
    ]

    for group in takeaways:
        for line in group:
            print(f"  {line}")
        print()   # Blank line between takeaways

    print("=" * 65)
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================
# if __name__ == "__main__": ensures this only runs when the script is
# called directly, not when imported by another module.
# C# analogy: static void Main(string[] args) in Program.cs
# =============================================================================

if __name__ == "__main__":
    run_demo()   # Run the full multi-tenant platform demonstration
