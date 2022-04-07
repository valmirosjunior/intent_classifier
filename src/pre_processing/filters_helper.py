__maximum_note_ask = "Por favor, antes de desconectar, avalie esse serviço.\nDê uma nota de 1 a 10, sendo:\n \n1.Ruim 😔 >>> 10. Muito bom! 🤩"
__questions_answered_ask = "Suas dúvidas foram respondidas?"


def is_maximum_note(x):
    conversation = list(x['txt'])
    if __maximum_note_ask in conversation:
        index_ask = conversation.index(__maximum_note_ask)

        if index_ask < len(conversation) - 1:
            if conversation[index_ask + 1] == '10':
                return True
    return False


def is_questions_answered(x):
    conversation = list(x['txt'])
    if __questions_answered_ask in conversation:
        index_ask = conversation.index(__questions_answered_ask)

        if index_ask < len(conversation) - 1:
            if conversation[index_ask + 1].lower() == 'sim':
                return True
    return False
