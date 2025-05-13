

import ast


from tqdm import tqdm


from ..outputs import NormPhrase
from ..configs import NormalizeConfig



class PhraseNormalizer:
    """
    Normalizes a list of text phrases using a language model.

    This class iterates through a list of phrases and uses an external API
    (e.g., OpenAI) to convert each phrase into a standardized or "cleaned" format
    based on a provided system prompt.

    Attributes:
        client (Any): The API client (e.g., OpenAI client) used for normalization.
        model_name (str): The name of the language model to use.
        temperature (float): The sampling temperature for the language model.
        print_log (bool): If True, prints the original and cleaned version of each phrase.
        display_pbar (bool): If True, displays a progress bar during normalization.
                             Cannot be True if print_log is also True.
        system_prompt (str): The system prompt used to instruct the language model
                             on how to normalize the phrases.
    """
    def __init__(
        self,
        norm_cfg:NormalizeConfig,
    ):
        """
        Initializes the PhraseNormalizer with configuration for the normalization process.

        Args:
            norm_cfg: An instance of NormalizeConfig containing the API client,
                      model name, temperature, logging/progress bar flags, and
                      system prompt.
        """
        self.client=norm_cfg.client
        self.model_name=norm_cfg.model_name
        self.temperature=norm_cfg.temperature
        self.print_log=norm_cfg.print_log
        self.display_pbar=norm_cfg.display_pbar
        self.system_prompt=norm_cfg.system_prompt


    def normalize(
        self,
        phrases:list[str],
    ):
        """
        Normalizes a list of input phrases using the configured language model.

        Each phrase is sent to the language model individually for normalization.
        Progress can be tracked via a progress bar or console logs.

        Args:
            phrases: A list of string phrases to be normalized.

        Returns:
            A list of normalized phrase strings.
        """
        
        print_log=self.print_log
        display_pbar=self.display_pbar
        if print_log and display_pbar:
            print(
                f"print_log and display_pbar cannot be same value, setting print_log=False and display_pbar=True"
            )
            print_log=False
            display_pbar=True
        
        pbar=None
        if display_pbar:
            pbar = tqdm(
                colour="blue",
                desc="Normalizing Phrases",
                total=len(phrases),
                dynamic_ncols=True
            )
        cleaned_phrases=[]
        for phrase in phrases:
            messages = [
                { "role": "system" , "content": self.system_prompt},
                { "role": "user" , "content": f"Normalize the following scope:\n{phrase}"}
            ]
    
            resp=self.client.beta.chat.completions.parse(
                model = self.model_name, 
                messages = messages, 
                temperature = self.temperature, 
                response_format = NormPhrase
            )
            cleaned_phrase=NormPhrase(
                **ast.literal_eval(resp.choices[0].message.content)
            ).phrase
            if print_log:
                print(
                    f"Original: {phrase}\nCleaned: {cleaned_phrase}\n"
                )
            cleaned_phrases.append(cleaned_phrase)
            if pbar:
                pbar.update(1)
                pbar.set_description(
                    f"Phrases Processed: {len(cleaned_phrases)}/{len(phrases)}"
                )
        return cleaned_phrases






#