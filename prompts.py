"""
Prompts and text templates for the arXiv summarizer
"""

class Prompts:
    """Collection of prompts and text templates used throughout the application"""
    
    # OpenAI prompt for summarizing arXiv papers
    TWEET_SUMMARY_PROMPT = (
        "Craft a tweet-sized summary of this article. Start with a question to draw readers in, "
        "use a relatable analogy if possible, and end with a call to action. Explain the research "
        "findings in simple terms and highlight how they could impact everyday life. Include 3-4 "
        "targeted hashtags for better reach. "
    )
    
    # Enhanced engaging prompt with growth strategies
    ENGAGING_SUMMARY_PROMPT = (
        "Create a viral-worthy summary of this research paper. Follow this structure:\n"
        "1. Start with an attention-grabbing question or bold statement\n"
        "2. Summarize the key finding in simple terms with a relatable analogy\n"
        "3. Add a 'Why it matters:' sentence explaining real-world impact\n"
        "4. End with a mini-opinion or contrarian take that sparks discussion\n"
        "Do not make bold or italic style, just plain text. "
        "Make it shareable, not just informative. Add personality and perspective. "
    )
    
    # Personality-driven technical prompt
    TECHNICAL_WITH_PERSONALITY_PROMPT = (
        "Create a technical summary with personality for researchers and practitioners. "
        "Include methodology, key findings, and implications. Add a critical perspective or "
        "contrarian opinion at the end. What could go wrong? What are the limitations? "
        "Be skeptical but fair. Include relevant technical hashtags. "
    )
    
    # Contrarian/skeptical style prompt
    CONTRARIAN_PROMPT = (
        "Take a skeptical but fair approach to this research. Summarize the findings but also "
        "highlight potential limitations, overhyped claims, or questions that remain unanswered. "
        "What should we be cautious about? End with a thought-provoking question. "
    )
    
    # Twitter thread opener
    THREAD_OPENER = (
        "📚 Today's scientific breakthroughs could change everything! Dive into these handpicked "
        "discoveries. Knowledge is power! 👇 "
    )
    
    # Twitter thread closer
    THREAD_CLOSER = (
        "🚀 That's a wrap on today's scientific wonders! Follow my account to stay updated on the "
        "latest discoveries that could shape tomorrow. "
    )
    
    # Personality-driven thread openers
    VIRAL_THREAD_OPENER = (
        "🔥 Today's research could break the internet (or at least your assumptions)! "
        "Buckle up for discoveries that'll make you question everything. Thread 👇 "
    )
    
    CONTRARIAN_THREAD_OPENER = (
        "📊 Time to burst some research bubbles! Today's papers look impressive on the surface, "
        "but let's dig deeper into what they REALLY mean. Skeptical thread 👇 "
    )
    
    PERSONALITY_THREAD_CLOSER = (
        "💭 Hot take: Half of these papers won't replicate, but the other half might change your life. "
        "Follow for more research reality checks and the occasional gem! "
    )

class PromptTemplates:
    """Dynamic prompt templates that can be customized"""
    
    @staticmethod
    def get_summary_prompt(paper_text: str, style: str = "engaging") -> str:
        """
        Generate a summary prompt for a given paper text and style.
        
        Args:
            paper_text (str): The full text of the research paper
            style (str): The style of summary ('engaging', 'viral', 'technical', 'contrarian')
        
        Returns:
            str: The complete prompt including the paper text
        """
        if style == "engaging":
            base_prompt = Prompts.TWEET_SUMMARY_PROMPT
        elif style == "viral":
            base_prompt = Prompts.ENGAGING_SUMMARY_PROMPT
        elif style == "technical":
            base_prompt = Prompts.TECHNICAL_WITH_PERSONALITY_PROMPT
        elif style == "contrarian":
            base_prompt = Prompts.CONTRARIAN_PROMPT
        
        return base_prompt + paper_text
    
    @staticmethod
    def get_thread_opener(topic: str = None, personality: str = "default") -> str:
        """
        Get a thread opener, optionally customized for a specific topic and personality.
        
        Args:
            topic (str): Optional topic to customize the opener
            personality (str): Style of opener ('default', 'viral', 'contrarian')
        
        Returns:
            str: The thread opener text
        """
        if personality == "viral":
            base_opener = Prompts.VIRAL_THREAD_OPENER
        elif personality == "contrarian":
            base_opener = Prompts.CONTRARIAN_THREAD_OPENER
        else:
            base_opener = Prompts.THREAD_OPENER
        
        if topic and personality == "default":
            return f"📚 Today's {topic} breakthroughs could change everything! Dive into these handpicked discoveries. Knowledge is power! 👇 "
        elif topic and personality == "viral":
            return f"🔥 {topic} research is about to blow your mind! These discoveries will make you question everything you thought you knew. Thread 👇 "
        elif topic and personality == "contrarian":
            return f"🤔 Let's fact-check today's {topic} hype! Time to separate breakthrough from buzzword. Skeptical thread 👇 "
        else:
            return base_opener
    
    @staticmethod
    def get_thread_closer(custom_message: str = None, personality: str = "viral") -> str:
        """
        Get a thread closer, optionally with a custom message or viral.
        
        Args:
            custom_message (str): Optional custom closing message
            personality (str): Style of closer ('default', 'viral')
        
        Returns:
            str: The thread closer text
        """
        if custom_message:
            return custom_message
        elif personality == "viral":
            return Prompts.PERSONALITY_THREAD_CLOSER
        else:
            return Prompts.THREAD_CLOSER

class PaperScoringPrompts:
    """Prompts for scoring and ranking research papers"""
    
    BATCH_SCORING_PROMPT = """
    You are an expert science communicator who helps non-technical audiences discover fascinating research. 
    Score each paper below on a scale of 1-10 for NON-TECHNICAL AUDIENCE APPEAL based on:
    
    1. ACCESSIBILITY (3 points): How easily can this be explained to someone without a PhD?
    2. REAL-WORLD IMPACT (3 points): Will this matter to people's daily lives or society?
    3. WOW FACTOR (2 points): Is this surprising, counterintuitive, or exciting enough to share?
    4. VISUAL POTENTIAL (2 points): Can we create engaging visuals/analogies from this?
    
    For each paper, provide EXACTLY this format (one line per paper):
    Paper N: Score=X, Reason="brief reason", Area="research field"
    
    Be concise and use simple quotes. Here are the papers:
    """
    
    DIVERSITY_FILTER_PROMPT = """
    From this list of high-scoring papers, select 5-8 papers that provide good diversity across:
    - Research areas (avoid too many AI papers, include physics, biology, etc.)
    - Impact types (theoretical breakthroughs, practical applications, surprising findings)
    - Complexity levels (mix of accessible and slightly more technical)
    
    Prioritize papers with scores 7+ but ensure variety. List the selected paper numbers only.
    
    Scored papers:
    """

class ViralContentTemplates:
    """Templates for creating viral, shareable content"""
    
    @staticmethod
    def generate_why_it_matters(research_summary: str) -> str:
        """
        Generate a 'Why it matters:' statement for research.
        
        Args:
            research_summary (str): Brief summary of the research
            
        Returns:
            str: Prompt to generate a why-it-matters statement
        """
        return (
            f"Based on this research summary: '{research_summary}'\n"
            "Generate a single compelling sentence starting with 'Why it matters:' that explains "
            "the real-world impact, business implications, or how this could affect people's daily lives. "
            "Make it concrete and specific, not vague. Examples:\n"
            "- 'Why it matters: makes billion-parameter models run on laptops, not just H100s.'\n"
            "- 'Why it matters: could kill small startups relying on expensive inference APIs.'\n"
            "- 'Why it matters: your smartphone could soon run GPT-4 level AI locally.'"
        )
    
    @staticmethod
    def generate_contrarian_take(research_summary: str) -> str:
        """
        Generate a contrarian or critical perspective on research.
        
        Args:
            research_summary (str): Brief summary of the research
            
        Returns:
            str: Prompt to generate a contrarian take
        """
        return (
            f"Based on this research: '{research_summary}'\n"
            "Generate a thoughtful but skeptical take. What could go wrong? What are the limitations? "
            "What questions remain unanswered? Be critical but fair. Examples:\n"
            "- 'But will it work outside carefully controlled lab conditions?'\n"
            "- 'Impressive results, but only tested on English - what about other languages?'\n"
            "- 'Sounds great until you consider the massive computational costs.'\n"
            "End with a thought-provoking question."
        )
    
    @staticmethod
    def suggest_visual_content(research_topic: str) -> str:
        """
        Suggest visual content ideas for a research topic.
        
        Args:
            research_topic (str): The main topic/field of research
            
        Returns:
            str: Visual content suggestions
        """
        return (
            f"For research topic '{research_topic}', suggest 3 simple visual content ideas:\n"
            "1. A before/after comparison chart\n"
            "2. A simple infographic showing the key numbers\n"
            "3. A meme or diagram that explains the concept\n"
            "Keep visuals simple, shareable, and understandable at a glance."
        )
    
    ENGAGEMENT_HOOKS = [
        "🚨 Plot twist:",
        "💡 Here's the thing:",
        "🔥 Hot take:",
        "📊 Reality check:",
        "⚡ Breaking:",
        "🎯 Bottom line:",
        "🤯 Mind-bending fact:",
        "💭 Unpopular opinion:",
        "🌟 Game changer:",
        "⚠️ Heads up:",
        "🔍 Food for thought:",
        "🧠 Think about this:",
    ]
    
    DISCUSSION_STARTERS = [
        "What could go wrong here?",
        "Too good to be true?",
        "Who benefits most from this?",
        "What are we missing?",
        "Overhyped or undervalued?",
        "Will this actually scale?",
        "What would you use this for?",
        "Thoughts? Am I being too skeptical?",
        "Who's this bad news for?",
        "Revolutionary or incremental?",
        "What’s the next step for this research?",
        "How could this change daily life?",
        "What’s the biggest limitation here?",
        "Could this backfire in some way?",
        "How does this compare to existing solutions?",
        "What’s the ethical angle here?",
        "What’s the most exciting potential application?",
    ]

class MessageTemplates:
    """Templates for various status and error messages"""
    
    # Status messages
    PROCESSING_PAPER = "Processing: {title}"
    EXTRACTING_AUTHORS = "Authors: {authors}"
    ARXIV_ID_EXTRACTED = "Extracted arXiv ID: {arxiv_id}"
    AUTHOR_LOOKUP_COMPLETE = "Author lookup complete: {found}/{total} Twitter handles found"
    
    # Author tagging messages
    SEMANTIC_SCHOLAR_SEARCH = "Searching Semantic Scholar for author social links..."
    AUTHOR_PROGRESS = "[{current}/{total}] Looking up Twitter handle for: {author_name}"
    FOUND_VIA_SEMANTIC_SCHOLAR = "  ✓ Found via Semantic Scholar homepage: @{handle}"
    FOUND_VIA_SEMANTIC_SCHOLAR_DETAILED = "  ✓ Found via Semantic Scholar detailed lookup: @{handle}"
    FOUND_VIA_ORCID = "  ✓ Found via ORCID: @{handle}"
    NO_HANDLE_FOUND = "  ✗ No Twitter handle found for {author_name}"
    CHECKING_DETAILED_PROFILE = "  → Checking detailed Semantic Scholar profile..."
    TRYING_ORCID_FALLBACK = "  → Semantic Scholar didn't find handle, trying ORCID..."
    PROGRESS_UPDATE = "  Progress: {found}/{processed} handles found so far"
    
    # API messages
    RATE_LIMITED_SEMANTIC_SCHOLAR = "Rate limited by Semantic Scholar API, waiting {seconds} seconds..."
    RATE_LIMITED_AUTHOR_LOOKUP = "Rate limited on author lookup, waiting {seconds} seconds..."
    SEMANTIC_SCHOLAR_FAILED = "Semantic Scholar API failed with status {status_code}"
    ORCID_SEARCH_FAILED = "ORCID search failed with status {status_code}"
    
    # Image processing messages
    IMAGE_EXTRACTED = "Image extracted and saved to: {path}"
    RELEVANT_IMAGE_EXTRACTED = "Relevant image extracted and saved to: {path}"
    NO_IMAGES_FOUND = "No images found in PDF."
    NO_RELEVANT_IMAGES = "No relevant images found in PDF."
    GIF_CREATED = "GIF created and saved to: {path}"
    
    # General status messages
    RESULTS_COUNT = "Number of results: {count}"
    NO_RESULTS_FOUND = "No results found for the specified date range. Please check your query and try again."
    TWITTER_AUTH_SUCCESS = "Successfully authenticated to Twitter."
    
    @staticmethod
    def format_message(template: str, **kwargs) -> str:
        """Format a message template with the provided arguments"""
        return template.format(**kwargs)
