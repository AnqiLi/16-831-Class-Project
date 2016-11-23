classdef UCBPolicy < Policy
    %POLICYUCB This is a concrete class implementing UCB.

    properties
        % Member variables
        nSites
        sumSatisf
        sumWaitTime
        countObserved
        round
        lastAction
        ubSatisfs
        lbWaitTimes
        lastUb
        alpha
        siteDists
        game
        weightDist           % weight on distance
        weightWait          % weight on wait time
        weightRide           % weight on ride satisfaction
    end
    
    methods
        function self = UCBPolicy(game)
            % Initialize
            self.nSites = game.nSites;
            self.round = 0;
            self.siteDists = game.siteDist;
            self.weightDist = game.weightDist;
            self.weightWait = game.weightWait;
            self.weightRide = game.weightRide;
            self.game = game;
            
            self.sumSatisf = zeros(1, self.nSites);
            self.sumWaitTime = zeros(1, self.nSites);
            self.countObserved = 1e-5*ones(1, self.nSites);
            self.alpha = 1;
        end
        
        function action = decision(self, site, ~)
            % Choose action
            self.round = self.round + 1;
            C = self.countObserved;
            ubSatisf = self.sumSatisf./C + sqrt(self.alpha*log(self.round)./(2*C));
            lbWaitTime = max(self.sumWaitTime./C - sqrt(self.alpha*log(self.round)./(2*C)),0);
            if self.round == 1
                dist = self.game.m0*ones(1, self.nSites);
            else
                dist = self.siteDists(site,:);
            end
            self.ubSatisfs = [self.ubSatisfs; ubSatisf];
            self.lbWaitTimes = [self.lbWaitTimes; lbWaitTime];
            
            if site > 0
                ubSatisf(site) = 0;
            end

            ubReward = -self.weightDist*dist - self.weightWait*lbWaitTime + self.weightRide*ubSatisf;
            [~, action] = max(ubReward);


                      
        end
        
        function updatePolicy(self, prevsite, site, satisf, waittime)
            % Update ucb

            if prevsite ~= site
                self.sumWaitTime(site) = self.sumWaitTime(site) + waittime;
                self.countObserved(site) = floor(self.countObserved(site) + 1); 
                self.sumSatisf(site) = self.sumSatisf(site) + satisf;
            end

        end        
        
        function drawUpperBounds(self)
            figure(3); clf; hold on;
            for i = 1:self.nSites
                plot(self.ubSatisfs(:, i));
            end
            plot([1,1,1,1], self.game.means, 'o');
            legend('1','2','3','4');
            xlabel('rounds'); ylabel('upper bound');
            title('Upper bound on ride satisfaction')
            hold off;
            
            figure(4); clf; hold on;
            for i = 1:self.nSites
                plot(self.lbWaitTimes(:, i));
            end
            legend('1','2','3','4');
            xlabel('rounds'); ylabel('lower bound');
            title('Lower bound on waittime')
            hold off;
            
            
        end
    end

end
