from models import EncoderRNN, DecoderRNN, Seq2seq

enc = EncoderRNN()
dec = DecoderRNN()
model = Seq2seq(enc, dec)
