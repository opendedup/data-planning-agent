"""
Interactive CLI for Planning Agent

Provides a command-line interface for testing the planning conversation flow.
"""

import asyncio
import logging
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from ..clients.gemini_client import GeminiClient
from ..clients.storage_client import StorageClient
from ..core.conversation import ConversationManager
from ..core.prp_generator import PRPGenerator
from ..core.refiner import RequirementRefiner
from ..mcp.config import load_config

logger = logging.getLogger(__name__)
console = Console()


class InteractiveCLI:
    """
    Interactive command-line interface for planning sessions.
    """

    def __init__(self):
        """Initialize the interactive CLI."""
        # Ensure environment variables are loaded before config
        from dotenv import load_dotenv
        load_dotenv()
        
        # Load configuration
        self.config = load_config()

        # Load organizational context if configured
        from ..clients.context_loader import load_context_from_directory

        context = load_context_from_directory(self.config.context_dir)

        # Initialize Vertex AI Search client for data catalog queries
        vertex_search_client = None
        if self.config.vertex_project_id and self.config.vertex_datastore_id:
            from ..clients.vertex_search_client import VertexSearchClient
            
            vertex_search_client = VertexSearchClient(
                project_id=self.config.vertex_project_id,
                location=self.config.vertex_datastore_location,
                datastore_id=self.config.vertex_datastore_id,
            )

        # Initialize clients
        self.gemini_client = GeminiClient(
            api_key=self.config.gemini_api_key,
            model_name=self.config.gemini_model,
            temperature=0.7,
            context=context,
            vertex_search_client=vertex_search_client,
            enable_reflection=self.config.enable_question_reflection,
        )
        self.storage_client = StorageClient(default_output_dir=self.config.output_dir)

        # Initialize conversation manager
        self.conversation_manager = ConversationManager()

        # Initialize refiner
        self.refiner = RequirementRefiner(
            gemini_client=self.gemini_client,
            conversation_manager=self.conversation_manager,
            max_turns=self.config.max_conversation_turns,
        )

        # Initialize PRP generator
        self.prp_generator = PRPGenerator(
            gemini_client=self.gemini_client, storage_client=self.storage_client
        )

    def display_welcome(self) -> None:
        """Display welcome message."""
        welcome_text = """
# Data Planning Agent - Interactive Mode

Welcome! I'll help you create a comprehensive **Data Product Requirement Prompt** (Data PRP)
through a conversational process.

## How it works:

1. 📝 Tell me your initial business intent or goal
2. 💬 I'll ask clarifying questions (mostly multiple choice)
3. ✅ Once I have enough information, we'll generate your Data PRP
4. 💾 The final document will be saved for use with the Data Discovery Agent

Let's get started!
        """
        console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="blue"))
        console.print()

    async def run(self) -> None:
        """Run the interactive planning session."""
        try:
            # Display welcome
            self.display_welcome()

            # Get initial intent
            console.print("[bold cyan]Step 1:[/bold cyan] What is your business intent or goal?")
            console.print(
                "[dim]Example: We want to provide the merchandising team insights into trending items[/dim]"
            )
            console.print()

            initial_intent = Prompt.ask("Your business intent")

            if not initial_intent.strip():
                console.print("[red]❌ Intent cannot be empty. Exiting.[/red]")
                return

            console.print()
            console.print("[yellow]⏳ Starting planning session...[/yellow]")
            console.print()

            # Start session
            session_id, questions = await self.refiner.start_session(initial_intent)

            console.print(
                Panel(
                    f"[green]✨ Session Started![/green]\n\nSession ID: [cyan]{session_id}[/cyan]",
                    border_style="green",
                )
            )
            console.print()

            # Conversation loop
            is_complete = False
            turn_count = 0

            while not is_complete and turn_count < self.config.max_conversation_turns:
                # Display questions
                console.print(Panel(Markdown(questions), title="Questions", border_style="cyan"))
                console.print()

                # Get user response
                console.print("[bold cyan]Your response:[/bold cyan]")
                console.print(
                    "[dim]Tip: For multiple choice, include the letter (a, b, c, d) and any additional details[/dim]"
                )
                console.print()

                user_response = Prompt.ask("Response")

                if not user_response.strip():
                    console.print("[red]Response cannot be empty. Please try again.[/red]")
                    console.print()
                    continue

                console.print()
                console.print("[yellow]⏳ Processing your response...[/yellow]")
                console.print()

                # Continue conversation
                next_questions, is_complete = await self.refiner.continue_conversation(
                    session_id, user_response
                )

                questions = next_questions
                turn_count += 1

            # Check completion status
            if is_complete:
                console.print(
                    Panel(
                        "[green]✅ Requirements gathering complete![/green]\n\n"
                        "I believe I have enough information to generate your Data PRP.",
                        border_style="green",
                    )
                )
                console.print()
                
                # Show conversation summary
                session = self.refiner.get_session(session_id)
                console.print("[bold]Conversation Summary:[/bold]")
                console.print(Panel(session.get_conversation_text(), border_style="dim"))
                console.print()
                
                # Ask for user confirmation
                console.print("[bold cyan]Confirmation:[/bold cyan]")
                console.print(
                    "[dim]This is your last chance to add important details or corrections.[/dim]"
                )
                console.print()
                
                proceed = Prompt.ask(
                    "Shall I proceed with generating the final requirement prompt?",
                    choices=["y", "n"],
                    default="y"
                )
                
                if proceed.lower() == "n":
                    console.print()
                    console.print("[yellow]Let's gather more details...[/yellow]")
                    console.print()
                    console.print("[bold cyan]What additional information would you like to add?[/bold cyan]")
                    console.print()
                    
                    additional_info = Prompt.ask("Additional details")
                    
                    if additional_info.strip():
                        console.print()
                        console.print("[yellow]⏳ Processing additional information...[/yellow]")
                        console.print()
                        
                        # Continue conversation with additional info
                        next_questions, is_complete = await self.refiner.continue_conversation(
                            session_id, additional_info
                        )
                        
                        if is_complete:
                            console.print(
                                Panel(
                                    "[green]✅ Additional information captured![/green]\n\n"
                                    "Ready to generate your Data PRP.",
                                    border_style="green",
                                )
                            )
                        else:
                            console.print(Panel(Markdown(next_questions), title="Follow-up", border_style="cyan"))
                            console.print()
                            console.print("[yellow]Please continue the conversation above, then we'll proceed to generation.[/yellow]")
                            return
                    
                console.print()
            else:
                console.print(
                    Panel(
                        "[yellow]⚠️ Maximum conversation turns reached.[/yellow]\n\n"
                        "Proceeding to generate Data PRP with available information.",
                        border_style="yellow",
                    )
                )
            console.print()

            # Ask about output path
            console.print("[bold cyan]Step 2:[/bold cyan] Where should I save the Data PRP?")
            console.print(
                f"[dim]Default: {self.config.output_dir} (press Enter to use default)[/dim]"
            )
            console.print("[dim]Or provide a custom path (GCS: gs://bucket/path or local path)[/dim]")
            console.print()

            output_path = Prompt.ask("Output path", default="")
            output_path = output_path.strip() if output_path else None

            console.print()
            console.print("[yellow]⏳ Generating Data Product Requirement Prompt...[/yellow]")
            console.print()

            # Get session
            session = self.refiner.get_session(session_id)

            # Generate PRP
            prp_content, file_path = await self.prp_generator.generate_prp(
                session=session, output_path=output_path, save_to_file=True
            )

            # Display success
            console.print(
                Panel(
                    f"[green]✅ Data PRP Generated Successfully![/green]\n\n"
                    f"📄 Saved to: [cyan]{file_path}[/cyan]",
                    border_style="green",
                )
            )
            console.print()

            # Display preview
            console.print("[bold]Preview of your Data PRP:[/bold]")
            console.print()
            console.print(Panel(Markdown(prp_content), border_style="blue"))
            console.print()

            console.print("[green]✨ All done! You can now use this Data PRP with the Data Discovery Agent.[/green]")

        except KeyboardInterrupt:
            console.print()
            console.print("[yellow]Session interrupted by user. Goodbye![/yellow]")
            sys.exit(0)
        except ValueError as e:
            # User-friendly errors (like safety filter blocks)
            console.print()
            console.print(f"[yellow]⚠️  {str(e)}[/yellow]")
            console.print()
            console.print("[dim]Tip: Try rephrasing your input to be more general and avoid:[/dim]")
            console.print("[dim]  • Specific names of people or organizations[/dim]")
            console.print("[dim]  • Gambling or betting-related topics[/dim]")
            console.print("[dim]  • Potentially controversial subjects[/dim]")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error in interactive CLI: {e}", exc_info=True)
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            sys.exit(1)


def main() -> None:
    """Main entry point for the interactive CLI."""
    # Load environment variables first
    from dotenv import load_dotenv
    load_dotenv()
    
    # Set up logging
    import os
    log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run interactive CLI
    cli = InteractiveCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()

