from jinja2 import Environment, FileSystemLoader

class Assistant:
    def __init__(self, system_prompt: str, template_path="templates/chat_template.jinja"):
        self.system_prompt = system_prompt
        self.env = Environment(loader=FileSystemLoader("templates"))
        self.template = self.env.get_template("chat_template.jinja")
        self.history = []

    def build_input(self, user_prompt: str, schema_prompt: str = ""):
        """
        Combine system, schema, and user messages into structured chat format.
        """
        messages = []

        # System / identity
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Optional "assistant" tone/schema instructions
        if schema_prompt:
            messages.append({"role": "assistant", "content": schema_prompt})

        # Add history (optional)
        messages.extend(self.history)

        # New user message
        messages.append({"role": "user", "content": user_prompt})

        rendered = self.template.render(
            messages=messages,
            add_generation_prompt=True,
            bos_token=""
        )
        return rendered

