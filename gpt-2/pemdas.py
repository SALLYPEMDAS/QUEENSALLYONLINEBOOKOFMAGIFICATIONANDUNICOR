#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import random
import tensorflow as tf
import model, sample, encoder
from datetime import datetime
import time
import calendar

def fortminute(oldtime = datetime.utcnow().__format__('%Y%m%d%H%M%S')):
    yyyymmdd = oldtime[:8]
    hhmmss = oldtime[8:]
    hr = hhmmss[:2]
    min = hhmmss[2:4]
    sec = hhmmss[4:]
    hr = int(hr)
    min  = int(min)
    sec = int(sec)
    seconds = (hr * (60 * 60)) + (min * 60) + sec

    return int((seconds/864)) - 1

def _seed(time = datetime.utcnow().__format__('%Y%m%d%H%M%S')):
    return time[:8] + str(fortminute(time))

def interact_model(
    model_name='124M',
    seed=int(_seed()),
    nsamples=8,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=50,
    top_p=0.93,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    print(seed)
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = random.randint(9, 93)
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

 #       print('seed: ' + str(seed))
 #       

        filename = str(seed) + str(calendar.timegm(time.gmtime())) + ".txt"
      
        o = open("../TXT/TXZ/GEN/" + filename, 'w')
        
        k = 0
        i = 0
        count = 0
        
        while True:
            raw_text = input("> ")
            while not raw_text:
                raw_text = input("> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            o.write(raw_text + "\n\n")

            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print(text)
                    o.write(text + "\n\n")
                    o.write("---\n\n")
                    o.flush()
            o.write("-------\n\n\n\n")
            o.flush()
#        with open("kjv.txt") as f:
#            for line in f:
#                print(line)
#                if (len(line) == 1):
#                    continue
#                prompt = "> " + filterline(line) + "\n\n"
#                context_tokens = enc.encode(prompt)
#                o = sess.run(output, feed_dict={
#                    context: [context_tokens for _ in range(batch_size)]
#                })[:, len(context_tokens):]
#                for i in range(batch_size):


        o.close()



if __name__ == '__main__':
    fire.Fire(interact_model)


