import tensorflow as tf 


def build_model(vocab_size, gru=False, rnn_units=256, additive_attention=True):
    #LAYERS DEFINITION
    #encoder & decoder inputs
    encoder_input = tf.keras.Input(shape=(None,))
    decoder_input = tf.keras.Input(shape=(None,))

    #attention
    if additive_attention:
        attention = tf.keras.layers.AdditiveAttention()
    else:
        attention = tf.keras.layers.Attention()

    #encoder & decoder embedding layer (same for both)
    e_embedding = tf.keras.layers.Embedding(vocab_size, rnn_units, mask_zero=True)
    d_embedding = tf.keras.layers.Embedding(vocab_size, rnn_units, mask_zero=True)

    #encoder & decoder
    if gru:
        e_rnn = tf.keras.layers.GRU(rnn_units, return_state=True)
        d_rnn = tf.keras.layers.GRU(rnn_units * 2, return_sequences=True, return_state=True)
    else:
        e_rnn = tf.keras.layers.LSTM(rnn_units, return_state=True)
        d_rnn = tf.keras.layers.LSTM(rnn_units * 2, return_sequences=True, return_state=True)


    #decoder Dense output layer
    d_dense_1 = tf.keras.layers.Dense(1024, activation="tanh")
    d_dense_2 = tf.keras.layers.Dense(vocab_size, activation="softmax")

    #MODEL DEFINITION
    #encoder def
    encoder = e_embedding(encoder_input)
    if gru:
        encoder_outs, f_encoder_h, b_encoder_h = tf.keras.layers.Bidirectional(e_rnn)(encoder)
        encoder_h = tf.keras.layers.concatenate([f_encoder_h, b_encoder_h])
    else:
        encoder_outs, f_encoder_h, f_encoder_c, b_encoder_h, b_encoder_c = tf.keras.layers.Bidirectional(e_rnn)(encoder)
        encoder_h = tf.keras.layers.concatenate([f_encoder_h, b_encoder_h])
        encoder_c = tf.keras.layers.concatenate([f_encoder_c, b_encoder_c])
    
    #decoder def
    decoder = d_embedding(decoder_input)
    if gru:
        decoder_outs, _ = d_rnn(decoder, initial_state=encoder_h)
    else:
        decoder_outs, _, _ = d_rnn(decoder, initial_state=[encoder_h, encoder_c])

    #attenrion 
    attention = tf.keras.layers.AdditiveAttention()([decoder_outs, encoder_outs])
    context_combined = tf.keras.layers.concatenate([attention, decoder_outs])

    decoder_output = d_dense_1(context_combined)
    decoder_output = tf.keras.layers.Dropout(0.25)(decoder_output)
    decoder_output = d_dense_2(decoder_output)

    #encoder-decoder model
    model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])

    #encoder for inference
    if gru:
        encoder_inf_model = tf.keras.Model(encoder_input, outputs=encoder_h)
    else:
        encoder_inf_model = tf.keras.Model(encoder_input, outputs=[encoder_h, encoder_c])

    #decoder for inference GRU
    if gru:
        decoder_inf_input_h = tf.keras.Input(shape=(rnn_units * 2))
        decoder_inf_initial_states = [decoder_inf_input_h]
        #decoder embedding
        decoder_emb = d_embedding(decoder_input)
        #decoder_inf 
        decoder_inf_outs, decoder_inf_h = d_rnn(decoder_emb, initial_state=decoder_inf_initial_states)
        decoder_inf_states = [decoder_inf_h]
        #decoder attention 
        attention_inf = tf.keras.layers.AdditiveAttention(causal=True)([decoder_inf_outs, decoder_inf_input_h])
        context_combined_inf = tf.keras.layers.Concatenate()([attention_inf, decoder_inf_outs])
        #decoder_inf dense
        decoder_inf_output = tf.keras.layers.TimeDistributed(d_dense_1)(context_combined_inf)
        decoder_inf_output = tf.keras.layers.TimeDistributed(d_dense_2)(decoder_inf_output)

        decoder_inf_model = tf.keras.Model([decoder_input] + decoder_inf_initial_states, [decoder_inf_output] + decoder_inf_states)

    #decoder for inference LSTM
    if not gru:
        decoder_inf_input_h = tf.keras.Input(shape=(rnn_units * 2))
        decoder_inf_input_c = tf.keras.Input(shape=(rnn_units * 2))
        decoder_inf_initial_states = [decoder_inf_input_h, decoder_inf_input_c]
        #decoder embedding
        decoder_emb = d_embedding(decoder_input)
        #decoder_inf 
        decoder_inf_outs, decoder_inf_h, decoder_inf_c = d_rnn(decoder_emb, initial_state=decoder_inf_initial_states)
        decoder_inf_states = [decoder_inf_h, decoder_inf_c]
        #decoder attention 
        attention_inf = tf.keras.layers.AdditiveAttention(causal=True)([decoder_inf_outs, decoder_inf_input_h])
        context_combined_inf = tf.keras.layers.Concatenate()([attention_inf, decoder_inf_outs])
        #decoder_inf dense
        decoder_inf_output = tf.keras.layers.TimeDistributed(d_dense_1)(context_combined_inf)
        decoder_inf_output = tf.keras.layers.Dropout(0.25)(decoder_inf_output)
        decoder_inf_output = tf.keras.layers.TimeDistributed(d_dense_2)(decoder_inf_output)

        decoder_inf_model = tf.keras.Model([decoder_input] + decoder_inf_initial_states, [decoder_inf_output] + decoder_inf_states)

    return model, encoder_inf_model, decoder_inf_model
