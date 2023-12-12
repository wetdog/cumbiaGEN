from music21 import converter, note

def xml_to_melody(filepath):

    stream = converter.parse(filepath)
    melody = [f"{note.pitch.name}{note.pitch.octave}-{note.duration.quarterLength}" for note in stream.flat.notes]
    return melody