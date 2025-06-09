import random

class ai4artsed_random_language_selector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }

    RETURN_TYPES = tuple(["STRING"] * 12)
    RETURN_NAMES = tuple([f"language_{i+1}" for i in range(12)])
    FUNCTION = "select_languages"
    CATEGORY = "AI4ArtsEd"

    def select_languages(self):
        supported_languages = [
            "English", "French", "German", "Spanish", "Italian", "Portuguese",
            "Dutch", "Russian", "Chinese", "Japanese", "Korean", "Arabic",
            "Hindi", "Bengali", "Turkish", "Vietnamese", "Polish", "Ukrainian",
            "Romanian", "Czech", "Greek", "Hungarian", "Swedish", "Finnish",
            "Norwegian", "Danish", "Hebrew", "Thai", "Indonesian", "Malay",
            "Yoruba", "Kiswahili", "Zulu", "Hausa", "Igbo", "Amharic",
            "Somali", "Afrikaans", "Xhosa", "Sesotho", "Twi", "Shona",
            "Kinyarwanda", "Luganda", "Wolof", "Fula", "Bambara", "Ewe",
            "Lakota", "Māori", "Quechua", "Aymara", "Guarani", "Mapudungun"
        ]
        selected_languages = random.sample(supported_languages, 12)
        return tuple(selected_languages)
