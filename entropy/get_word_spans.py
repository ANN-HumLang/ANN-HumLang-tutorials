def ajust_tokens_spans(self, caption):
        tok_cap = self.tokenizer(caption, add_special_tokens= False, is_split_into_words=True, return_offsets_mapping = True)
        sent = self.tokenizer.convert_ids_to_tokens(tok_cap['input_ids'])
        if len(caption) != len(tok_cap['input_ids']):
            #is_subword = np.array(tok_cap['offset_mapping'])[:,0] != 0
            is_subword = np.array(["##" in token for token in sent])
            subword_spans = []
            end = -1
            for i in reversed(range(len(is_subword))):
                if end > 0:
                    subword_spans.append([i, end])
                if is_subword[i] and end == -1:
                    end = i
                if not is_subword[i]:
                    end = -1
            subword_spans = sorted(subword_spans, key=itemgetter(1))
            mapping = [[x,x] for x in range(len(caption))]
            inc = 0
            diff = 0
            for j in range(len(is_subword)):
                if is_subword[j]:
                    inc += 1
                    diff += 1  
                else:
                    for m in range((j-diff),len(mapping)):
                        mapping[m][1] += inc
                    inc = 0          
            for [b,e] in span:
                subword_spans.append([mapping[b][1], mapping[e][1]])
            span = subword_spans
        caption = self.tokenizer.convert_ids_to_tokens(tok_cap['input_ids'])
        return caption, span, tok_cap['input_ids']