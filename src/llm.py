import os
import time
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from dotenv import load_dotenv
from typing import Generator

load_dotenv()

client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1"
)


class LLMError(Exception):
    pass


class LLMConnectionError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


def ask_llm(
    user_message: str, 
    system_prompt: str = "You are a helpful assistant.",
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> str:
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1024,
                temperature=0.7,
                timeout=30.0
            )
            return response.choices[0].message.content
        
        except APITimeoutError as e:
            last_exception = e
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (2 ** attempt)
                time.sleep(sleep_time)
                continue
            raise LLMConnectionError("Request timed out after retries. Please try again.")
        
        except RateLimitError as e:
            last_exception = e
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (2 ** attempt) * 2
                time.sleep(sleep_time)
                continue
            raise LLMRateLimitError("Rate limit exceeded. Please wait and try again.")
        
        except APIError as e:
            raise LLMError(f"API error: {e.message}")
        
        except Exception as e:
            raise LLMError(f"Unexpected error: {str(e)}")
    
    raise LLMError(f"Failed after {max_retries} attempts: {last_exception}")


def ask_llm_stream(
    user_message: str, 
    system_prompt: str = "You are a helpful assistant."
) -> Generator[str, None, None]:
    try:
        stream = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=True,
            timeout=30.0
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except APITimeoutError:
        yield "\n\n[Error: Request timed out]"
    
    except RateLimitError:
        yield "\n\n[Error: Rate limit exceeded]"
    
    except APIError as e:
        yield f"\n\n[Error: {e.message}]"
