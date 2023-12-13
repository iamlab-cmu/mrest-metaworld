import numpy as np

from metaworld.policies.policy import assert_fully_parsed
from metaworld.policies import *

BLOCKS = ['blockA', 'blockB', 'blockC']
OBJECTS = ['bread','bottle','coke','pepsi','milk', 'cereal', 'red_mug', 'white_mug', 'blue_mug',
        'reebok_shoe', 'reebok_blue_shoe', 'reebok_black_shoe', 'reebok_pink_shoe', 'pink_heel', 'brown_heel', 'green_shoe',
        'supplement0', 'supplement1', 'supplement2']
BIG_OBJECTS = ['drawer', 'door', 'window']

class SawyerMultitaskV2Policy():
    def __init__(self, skill='Pick and place', target_object='milk'):
        self.stage = 0

        self.policy = self.select_policy(skill, target_object)

    def select_policy(self, skill, target_object):
        
        skill_lower = skill.lower()
        # Skills for small blocks and objects
        if (target_object.startswith('block') or target_object in OBJECTS
            or target_object.startswith('stick')):
            if skill_lower.startswith('pick') and 'place' in skill_lower:
                policy = SawyerPickPlaceGenV2Policy(skill, target_object)
            elif skill_lower.startswith('pick'):
                policy = SawyerPickGenV2Policy(skill, target_object)
            elif 'push' in skill_lower and 'forward' in skill_lower:
                policy = SawyerPushForwardGenV2Policy(skill, target_object)
            elif 'push' in skill_lower and 'backward' in skill_lower:
                policy = SawyerPushBackwardGenV2Policy(skill, target_object)
            elif 'push' in skill_lower and 'left' in skill_lower:
                policy = SawyerPushLeftGenV2Policy(skill, target_object)
            elif 'push' in skill_lower and 'right' in skill_lower:
                policy = SawyerPushRightGenV2Policy(skill, target_object)
            elif 'reach' in skill_lower and 'forward' in skill_lower:
                # policy = SawyerReachForwardGenV2Policy(skill, target_object)
                pass
            elif 'reach' in skill_lower and 'backward' in skill_lower:
                # policy = SawyerReachBackwardGenV2Policy(skill, target_object)
                pass
            elif 'reach' in skill_lower and 'left' in skill_lower:
                # policy = SawyerReachLeftGenV2Policy(skill, target_object)
                pass
            elif 'reach' in skill_lower and 'right' in skill_lower:
                # policy = SawyerReachRightGenV2Policy(skill, target_object)
                pass
            elif 'reach' in skill_lower and 'above' in skill_lower:
                policy = SawyerReachAboveGenV2Policy(skill, target_object)
            elif 'stack' in skill_lower:
                policy = SawyerStackGenV2Policy(skill, target_object)
            elif 'put' in skill_lower and 'drawer' in skill_lower and 'open' in skill_lower:
                policy = SawyerPutBlockInOpenDrawerV2GenPolicy(skill, target_object)
            elif 'put' in skill_lower and 'drawer' in skill_lower:
                policy = SawyerPutBlockInDrawerV2GenPolicy(skill, target_object)
            elif 'push' in skill_lower and 'drawer' in skill_lower:
                policy = SawyerPushBlockInOpenDrawerV2GenPolicy(skill, target_object)
            elif skill_lower == 'binpick':
                # policy = SawyerBinPickGenV2Policy(skill, target_object)
                pass
            else:
                raise NotImplementedError('Skill not implemented')
        
        elif 'drawer' in target_object:
            if 'open' in skill_lower:
                policy = SawyerDrawerOpenV2GenPolicy(skill, target_object)
            elif 'close' in skill_lower:
                policy = SawyerDrawerCloseV2GenPolicy(skill, target_object)
        
        elif 'door' in target_object:
            if 'lock' in skill_lower:
                policy = SawyerDoorLockV2GenPolicy(skill, target_object)
            elif 'stick' in skill_lower and 'close' in skill_lower:
                policy = SawyerStickDoorCloseGenV2Policy(skill, target_object)
            elif 'open' in skill_lower:
                policy = SawyerDoorOpenGenV2Policy(skill, target_object)
            elif 'close' in skill_lower:
                policy = SawyerDoorCloseGenV2Policy(skill, target_object)
        
        elif 'window' in target_object:
            if 'open' in skill_lower:
                policy = SawyerWindowOpenGenV2Policy(skill, target_object)
            elif 'close' in skill_lower:
                policy = SawyerWindowCloseGenV2Policy(skill, target_object)
        
        elif 'peg' in target_object:
            if 'insert' in skill_lower:
                policy = SawyerPegInsertionGenV2Policy(skill, target_object)
        elif 'RoundNut' in target_object:
            if 'pick' in skill_lower:
                policy = SawyerPegPickGenV2Policy(skill, target_object)
            elif 'door' in skill_lower:
                policy = SawyerPutNutInDoorGenV2Policy(skill, target_object)

        elif 'faucet' in target_object:
            if 'rotate' in skill_lower:
                policy = SawyerFaucetRotateV2Policy(skill, target_object)

        elif 'stick' in target_object:
            if 'pick' in skill_lower and 'place' in skill_lower:
                policy = SawyerPickPlaceGenV2Policy(skill, target_object)
            elif skill_lower.startswith('pick'):
                policy = SawyerPickGenV2Policy(skill, target_object)
        return policy
    

    def get_action(self, obs):
        return self.policy.get_action(obs)