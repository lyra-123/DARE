class ImitationNet(nn.Module):
    def __init__(self,
                 C=8, k=8,
                 d_model=128, n_heads=4, n_layers=2,
                 dropout=0.1,
                 n_qp=5, n_skip=5, n_re=5):
        super().__init__()
        self.k = k
        self.C = C
        self.A_DIM = n_qp * n_skip * n_re

        d_bw = d_model // 4
        self.bw_encoder = nn.Sequential(
            nn.Linear(1, d_bw),
            nn.LayerNorm(d_bw),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_bw, d_bw),
        )

        d_dec = d_model // 4
        d_emb = max(8, d_dec // 3)
        self.qp_emb   = nn.Embedding(n_qp,   d_emb)
        self.skip_emb = nn.Embedding(n_skip,  d_emb)
        self.re_emb   = nn.Embedding(n_re,    d_emb)
        self.dec_proj = nn.Sequential(
            nn.Linear(d_emb * 3, d_dec),
            nn.ReLU(),
        )

        d_p1 = d_model // 4
        self.p1_hist_encoder = nn.Sequential(
            nn.Linear(C, d_p1),
            nn.LayerNorm(d_p1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_p1, d_p1),
        )

        d_diff = d_model // 4
        self.diff_hist_encoder = nn.Sequential(
            nn.Linear(C, d_diff),
            nn.LayerNorm(d_diff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_diff, d_diff),
        )

        self.hist_proj = nn.Sequential(
            nn.Linear(d_bw + d_dec + d_p1 + d_diff, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.p1_curr_encoder = nn.Sequential(
            nn.Linear(C, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 2),
        )
        self.diff_curr_encoder = nn.Sequential(
            nn.Linear(C, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 2),
        )
        self.curr_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        def make_head(n_cls):
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_cls),
            )

        self.head_qp   = make_head(n_qp)
        self.head_skip = make_head(n_skip)
        self.head_re   = make_head(n_re)
        # # ===== 5. 单动作策略头 =====
        # self.policy = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model // 2, self.A_DIM)
        # )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode_history(self, bw, dec, p1_hist, diff_hist):

        bw_feat   = self.bw_encoder(bw)

        qp_e      = self.qp_emb(dec[:, :, 0])
        skip_e    = self.skip_emb(dec[:, :, 1])
        re_e      = self.re_emb(dec[:, :, 2])
        dec_feat  = self.dec_proj(torch.cat([qp_e, skip_e, re_e], dim=-1))

        p1_feat   = self.p1_hist_encoder(p1_hist)
        diff_feat = self.diff_hist_encoder(diff_hist)

        x = torch.cat([bw_feat, dec_feat, p1_feat, diff_feat], dim=-1)
        x = self.hist_proj(x)
        x = self.transformer(x)
        return x[:, -1, :]

    def encode_current(self, p1_curr, diff_curr):
        p1_f   = self.p1_curr_encoder(p1_curr)
        diff_f = self.diff_curr_encoder(diff_curr)
        return self.curr_proj(torch.cat([p1_f, diff_f], dim=-1))

    def forward(self, encode_states, p1_states, diff_states):
        bw = encode_states[:, 0:1, :].permute(0, 2, 1)
        dec = encode_states[:, 1:4, :].permute(0, 2, 1).long()
        p1_hist = p1_states[:, :, :-1].permute(0, 2, 1)
        diff_hist = diff_states[:, :, :-1].permute(0, 2, 1)
        p1_curr = p1_states[:, :, -1]
        diff_curr = diff_states[:, :, -1]

        hist_ctx  = self.encode_history(bw, dec, p1_hist, diff_hist)
        curr_feat = self.encode_current(p1_curr, diff_curr)

        combined = torch.cat([hist_ctx, curr_feat], dim=-1)
        gate      = self.gate(combined)                        # (B, d_model)
        gated     = gate * hist_ctx + (1 - gate) * curr_feat
        fused     = self.fusion(torch.cat([gated, curr_feat], dim=-1))

        # # ===== 单动作 logits =====
        # logits = self.policy(fused)
        # actor = F.softmax(logits, dim=-1)
        # return actor, logits  # (B, num_cfg)
        return (
            self.head_qp(fused),
            self.head_skip(fused),
            self.head_re(fused),
        )