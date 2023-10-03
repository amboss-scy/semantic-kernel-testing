from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAITextEmbedding,
)
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore
import asyncio
import os
import semantic_kernel as sk
import shutil

# General settings
absolute_path = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.join(absolute_path, "./database")
api_key, org_id = sk.openai_settings_from_dot_env()

# Kernel setup
kernel = sk.Kernel()
kernel.add_chat_service(
    "chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)
)
kernel.add_text_embedding_generation_service(
    "ada", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id)
)
kernel.import_skill(sk.core_skills.TextMemorySkill())


# Functions
def reset_database():
    for fname in os.listdir(database_path):
        file_path = os.path.join(database_path, fname)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(
                f"Failed to reset database. {file_path} could not be deleted.\nReason: {e}\nConsider deleting folder database manually."
            )
    shutil.rmtree(database_path)


async def populate_memory() -> None:
    await kernel.memory.save_information_async(
        "de_articles",
        id="Allgemeine Histologie",
        text="Die Hauptgewebemasse des Körpers wird von Zellen gebildet, die sich vier verschiedenen Grundgewebearten zuordnen lassen: Binde-, Muskel-, Nerven- und Epithelgewebe. Die drei erstgenannten Formen werden im Zusammenhang mit ihren Funktionen in gesonderten Lerneinheiten behandelt. Als besonderer Bestandteil annähernd jedes Körperorgans soll im Folgenden das Epithelgewebe mit seinen Unterformen und Funktionen erklärt werden.",
    )
    await kernel.memory.save_information_async(
        "de_articles",
        id="Atemwege und Lunge",
        text="Die Lunge ist für die Atmung zuständig und besteht aus einem rechten und einem linken Lungenflügel. Beide Lungenflügel sind wiederum in Lappen unterteilt und von einem System aus luftleitenden Wegen (den Bronchien) durchzogen. Diese enden in sog. Lungenbläschen (den Alveolen), in denen der Gasaustausch stattfindet. Die gesamten unteren Atemwege mit Ausnahme der Alveolen sind mit zilientragendem Epithel ausgekleidet, das zur Immunabwehr und Reinigung der Lunge beiträgt. Die Alveolen hingegen tragen eine sehr dünne Epithelschicht, damit die Atemgase Sauerstoff und Kohlendioxid möglichst leicht in die umgebenen Lungenkapillaren diffundieren können. Die Lungenkapillaren gehören zum kleinen Blutkreislauf, der sauerstoffarmes Blut von der rechten Herzkammer zur Lunge und sauerstoffreiches Blut von den Lungenkapillaren zum linken Vorhof führt. Die Lunge selbst wird über ein zweites Gefäßsystem versorgt. Die Durchblutung der Lunge wird auch als Perfusion bezeichnet und steht in engem Verhältnis zur Ventilation – der Verteilung der Atemgase in der Lunge. Diese wird durch Druckunterschiede zwischen Alveolarraum und Außenwelt angetrieben, die u.a. von der Atemmuskulatur erzeugt werden.",
    )
    await kernel.memory.save_information_async(
        "de_articles",
        id="Aufbau und Funktionen der Blutgefäße",
        text="Die Gefäße des menschlichen Körpers haben zwei grundlegende Aufgaben: Einerseits sollen sie das Blut vom Herzen zu den Organen und zurück transportieren, andererseits ermöglichen sie in den Organen den Stoffaustausch zwischen den Zellen und dem Blut. Die größeren Gefäße wie Arterien und Venen dienen vor allem dem Transport des Blutes. Sie verzweigen sich, sobald sie ein Organ erreicht haben, zu einer Vielzahl kleiner Kapillaren und Venolen (sog. Endstrombahn). Dort findet der eigentliche Stoffaustausch statt.",
    )


async def search_memory() -> None:
    questions = [
        "Was ist die Lunge?",
        "Woraus besteht Gewebe?",
        "Welche Aufgaben haben Blutgefäße?",
    ]

    for question in questions:
        print(f"Question: {question}")
        result = await kernel.memory.search_async("de_articles", question)
        print(f"Answer: {result[0].text}\n")


# Program
reset_database()
kernel.register_memory_store(
    memory_store=ChromaMemoryStore(persist_directory=database_path)
)
asyncio.run(populate_memory())
asyncio.run(search_memory())

# summarize = kernel.create_semantic_function(
#     "{{$input}}\n\nOne line TLDR with the fewest words.",
#     max_tokens=50,
#     temperature=0.2,
#     top_p=0.5,
# )

# print(
#     summarize(
#         input(
#             "Type of paste text to summarize here and press Enter (no newline characters!): "
#         )
#     )
# )
