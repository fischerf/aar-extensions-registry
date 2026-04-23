import importlib

from agent.extensions.api import ExtensionAPI


def test_register_and_command_present() -> None:
    """
    Ensure the extension's `register(api)` populates the ExtensionAPI with an
    'inspect' slash-command entry.
    """
    api = ExtensionAPI(name="inspect_test")

    # Import the extension module and call its register() function
    mod = importlib.import_module("aar_ext_inspect")
    assert hasattr(mod, "register"), "Extension module must expose register(api)"
    register = getattr(mod, "register")
    register(api)

    # The extension should have registered a command named "inspect"
    assert "inspect" in api._commands, "Expected 'inspect' command to be registered"
    desc, fn = api._commands["inspect"]
    assert callable(fn), "Registered command handler must be callable"
    # Basic description sanity check (non-empty)
    assert isinstance(desc, str) and desc, "Command description should be a non-empty string"
