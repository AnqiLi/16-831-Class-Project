classdef EXP3DPPolicy < Policy
    %EXP3Policy This is a concrete class that implements the EXP3
    %           algorithm for our problem setting
    
    properties
        %Member variables
        nSites
        weights
        round
        siteDists
        game
        weightDist
        weightWait
        weightRide
    end
    
    methods
        function self = EXP3DPPolicy(game)
            
            self.nSites = game.nSites;
            self.round = 0;
            self.siteDists = game.siteDist;
            self.weightDist = game.weightDist;
            self.weightWait = game.weightWait;
            self.weightRide = game.weightRide;
            self.game = game;
            self.weights = ones(self.nSites,self.nSites);
            
        end
        
        function action = decision(self,site,~)         
            self.round = self.round + 1;
            if site == 0
                norm_wts = ones(1,self.nSites)/self.nSites;
            else
                norm_wts = self.weights(site,:)/sum(self.weights(site,:));
            end
            action_vect = mnrnd(1,norm_wts);
            [~,action] = find(action_vect,1);
%             self.lastAction = action;
        end
        
        function updatePolicy(self, prevsite, site, reward)
            if prevsite == 0
                return
            end
            norm_wts = self.weights(prevsite,:)/sum(self.weights(prevsite,:));        

            assert (reward <= 1 && reward >=0);
            lossScalar = 1-reward;
            
            lossVector = zeros(1,self.nSites);
            lossVector(site) = lossScalar / norm_wts(site);
            
            eta = sqrt(log(self.nSites)/(self.round*self.nSites));
            self.weights(prevsite,:) = self.weights(prevsite,:).*exp(-eta*lossVector);
        end
            
        
    end
    
end
