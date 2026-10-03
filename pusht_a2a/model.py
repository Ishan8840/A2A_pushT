from omegaconf import OmegaConf
from roboverse_learn.il.policies.a2a.a2a_policy import A2AImagePolicy
from roboverse_learn.il.utils.flow.flow_matchers import ConditionalFlowMatcher
from roboverse_learn.il.utils.vision.model_getter import get_resnet
from roboverse_learn.il.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder


def build_policy(cfg):
    if cfg.n_obs_steps % 8 or cfg.n_action_steps % 8:
        raise ValueError("Upstream CNN encoders require lengths divisible by 8")
    if cfg.horizon != cfg.n_obs_steps - 1 + cfg.n_action_steps:
        raise ValueError("horizon must equal history + future - 1")
    shape_meta = {"action": {"shape": [2]}, "obs": {
        "head_cam": {"shape": [3, 96, 96], "type": "rgb"},
        "agent_pos": {"shape": [2], "type": "low_dim"}}}
    encoder = MultiImageObsEncoder(shape_meta, get_resnet("resnet18", weights=None),
                                  use_group_norm=True, imagenet_norm=True)
    upstream = OmegaConf.load(__import__("pusht_a2a").UPSTREAM /
        "roboverse_learn/il/configs/policy_config/a2a.yaml")
    return A2AImagePolicy(shape_meta=shape_meta, obs_encoder=encoder,
        horizon=cfg.horizon, n_obs_steps=cfg.n_obs_steps, n_action_steps=cfg.n_action_steps,
        flow_net=upstream.flow_net,
        flow_matcher=ConditionalFlowMatcher(sigma=0, num_sampling_steps=cfg.flow_steps),
        action_ae=upstream.action_ae, latent_dim=cfg.latent_dim,
        decode_flow_latents=True, consistency_weight=upstream.consistency_weight,
        enc_contrastive_weight=upstream.enc_contrastive_weight,
        flow_contrastive_weight=upstream.flow_contrastive_weight)
