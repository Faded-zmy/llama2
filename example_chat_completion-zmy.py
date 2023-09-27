# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=8
    )

    dialogs = [
        [{"role": "user", "content": """
        The following is a conversation between User_A and User_B. Please help to extract information and summarize opinion on both sides from the following conversation.
    Requirement:
    1. Divide the information into two parts, a description of people and opinions on a topic.
    2. Summarize the conversation into one topic and the opinion of both sides. Give both sides's  way of talking and did he achieve a achievement such as convincing someone, getting a message when discuss the topic using a concise sentence or some words.
    3. For the description of people, construct an information card of both sides.
    4. "Todo" is what people is going to do, "Todo_Time" is corresponding time.
    5. If a certain key's information is not mentioned, fill in it with "None".
    6. Information can only be extracted from conversations, do not make up for it yourself.

    The structure of the information card is as follows:
    {"basic_information": {"Name": xxx, "Gender": xxx, "Date_of_Birth": xxx, "Place_of_Birth": xxx, "Race": xxx, "Nationality": xxx}, "background": {"Educational_Background": xxx, "Occupation": xxx, "Position": xxx, "Achievement": xxx}, "others": {"Personality": xxx, "Hobbies": xxx, "Good_at": xxx, "Not_good_at": xxx, "Topics_of_interest": xxx, "Topics_of_disinterest": xxx, "People_They_Admire": xxx, "People_They_Dislike": xxx, "Todo": xxx, "Todo_Time": xxx}}}

    The output format is as follows:
    User_A's information card: xxx

    User_B's information card: xxx

    Discussions:
    {"Topic": xxx,
    "Summary": xxx,
    "User_A's_opinion": xxx,
    "User_A's_way_of_talking": xxx,
    "User_A's_achievement":xxx,
    "User_B's_opinion": xxx,
    "User_B's_way_of_talking": xxx,
    "User_B's_achievement":xxx}

    The conversation is as follows:
    "User_A: Hi there! I couldn't help but notice your exceptional swordfighting skills. Can you tell me a bit about your background in that area?\n\nUser_B: Well, my background in swordfighting is quite unique. I received extensive training from the Faceless Men in the art of combat and assassination. So yes, I'm pretty handy with a sword.\n\nUser_A: Oh, the Faceless Men, that sounds intriguing. How did you end up training with them?\n\nUser_B: Oh, you know, just your typical journey of revenge and survival. I wanted to become an unstoppable force, so I sought out the Faceless Men in Braavos. They saw potential in me, and the rest is history.\n\nUser_A: That's quite the journey you've had. And I heard you were a highborn lady before all this. What made you choose the path of an assassin?\n\nUser_B: Yes, I was born into a noble family, surrounded by rules and expectations. But after witnessing the injustices of the world and the suffering of my loved ones, I realized that power and titles meant nothing without the ability to protect and seek justice. I decided to forge my own path and become a force to be reckoned with.\n\nUser_A: It must have been quite a change for you, going from a highborn lady to a trained assassin. What has been the biggest challenge in your journey?\n\nUser_B: The biggest challenge...trusting my instincts and learning to adapt. The world is filled with people who underestimate me because of my gender or background, but I refuse to let that define me. My journey has taught me to be quick-thinking and resourceful, always staying one step ahead. And let me tell you, it's a challenge that I'm happy to take on.\n\nUser_A: That's truly inspiring. I admire your determination and resilience. Is there anyone who has been a significant influence on your journey?\n\nUser_B: Well, there have been a few people who have shaped me into who I am today. My father, Ned Stark, taught me about honor and justice. My sister, Sansa, has shown incredible strength and resilience herself. And let's not forget Jaqen H'ghar, my mysterious mentor from the Faceless Men. They've all had their part to play in my evolution.\n\nUser_A: I see, those are some remarkable individuals in your life. On a different note, what do you do in your free time? Any hobbies or interests?\n\nUser_B: Oh, you know, the usual hobbies. Swordfighting, exploring dangerous places, picking up survival skills. You could say I have a knack for getting into trouble and finding my way out of it. Keeps life interesting, you know?\n\nUser_A: Definitely sounds like an adventurous life you lead. But tell me, what is one thing you're not so good at?\n\nUser_B: Well, I'm not exactly cut out for the political game. All that plotting and scheming, it's not really my style. I prefer a more direct approach, you know? But hey, who needs politics when you have a sharp sword and fierce determination?\n\nUser_A: Fair point, politics can be quite tiresome. What topics are you most interested in? What really gets you going?\n\nUser_B: Ah, let's see. I'm particularly interested in topics like revenge, justice, and the art of survival. When faced with the injustices of the world, I become a force to be reckoned with. But enough about me, what gets your blood pumping? Any deep-rooted passions?\n\nUser_A:\n   xxx\nUser_B:"
    """}]
        
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
