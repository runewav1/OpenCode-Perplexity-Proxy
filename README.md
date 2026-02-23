# OpenCode-Perplexity-Proxy
Localhost server that exposes the Perplexity API to utilize available third party provider models like Claude, GPT, Gemini. 

## Requirements:
OpenCode installed; Bun to run the server.

## Installation:
In "~/.config/opencode", create a folder called "pplx-proxy". In it, drop in server.ts. 

## Server Launch
Go to your shell of choice and run "bun run server.ts" to launch the server, which defaults to port 4099. 

## OpenCode Configuration
In your opencode.json file, define a custom OpenAI compatible provider (for example "perplexity providers"). Under the provider, define an _npm_ package "@ai-sdk/openai-compatible".
Then, set the display name for the provider; I use "PPLX", to differentiate from the existing Perplexity provider.
After that, set two options, the baseURL and the apiKey. The baseURL, assuming you didn't change the default host port, is "http://localhost:4099/v1". The apiKey is a placeholder, so just type in "proxy". 
Now, setup your models. You only need to define the internal ID using their model name, for example, "gpt-5-mini", and set their display name to the same thing (unless you want to display it differently). Optionally, you can set up variants (which I won't explain, but you can find documentation for in the OpenCode docs). After this, your configuration is all set. 

## Flow
Run "bun run server.ts" in the shell of your choice. **Keep that shell running** while you're using OpenCode. Start OpenCode, and navigate to your custom Perplexity provider, and choose a model. That's it, now you can use GPT, Gemini, and Claude models directly from Perplexity!

## Compatibility
Only tested on one Windows device. I suppose it _should_ work on Linux or MacOS, though it might need to be tweaked to work as intended. Tool calling works, as does subagent delegation. No context usage or cost breakdown is supported/built into OpenCode, and isn't supplied from the server endpoint, so it won't appear in the sidebar or title bar of your OpenCode instance when using models through this server. 
