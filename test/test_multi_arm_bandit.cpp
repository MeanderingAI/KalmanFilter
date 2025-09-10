#include <gtest/gtest.h>
#include <bandit_arm.h>
#include <epsilon_greedy_agent.h>
#include <ucb_agent.h>

// Test fixture for BanditArm
class BanditArmTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_bandit = new BanditArm(0.5); 
    }

    void TearDown() override {
        delete test_bandit;
    }

    BanditArm* test_bandit;
};

// Test case 1: Verify initial state
TEST_F(BanditArmTest, InitialStateIsCorrect) {
    ASSERT_EQ(test_bandit->get_estimated_prob(), 0.0);
    ASSERT_EQ(test_bandit->get_pull_count(), 0);
}

// Test case 2: Verify a single pull and update
TEST_F(BanditArmTest, SinglePullAndUpdate) {
    double reward = test_bandit->pull();
    test_bandit->update(reward);
    ASSERT_EQ(test_bandit->get_pull_count(), 1);
    ASSERT_EQ(test_bandit->get_estimated_prob(), reward);
}

// Test case 3: Verify multiple updates
TEST_F(BanditArmTest, MultipleUpdates) {
    test_bandit->update(1.0); 
    test_bandit->update(0.0); 
    test_bandit->update(1.0); 
    
    ASSERT_EQ(test_bandit->get_pull_count(), 3);
    ASSERT_NEAR(test_bandit->get_estimated_prob(), 2.0 / 3.0, 0.001); 
}

// Test case 1: Pure exploitation (epsilon = 0)
TEST(EpsilonGreedyAgentTest, ExploitationWorksCorrectly) {
    std::vector<double> true_probs = {0.2, 0.9, 0.5};
    EpsilonGreedyAgent agent(true_probs, 0.0, 2);
    
    // Manually perform a single pull on the best arm to establish it as the best choice.
    // This circumvents the initial state where all probabilities are 0.
    agent.get_bandit(1).update(1.0);
    
    // Simulate 9 more pulls
    agent.run_simulation(9);
    
    // With pure exploitation, the agent should now always choose the best arm (index 1)
    ASSERT_EQ(agent.get_bandit(0).get_pull_count(), 0);
    ASSERT_EQ(agent.get_bandit(1).get_pull_count(), 10);
    ASSERT_EQ(agent.get_bandit(2).get_pull_count(), 0);
}
// Test case 2: Pure exploration (epsilon = 1)
TEST(EpsilonGreedyAgentTest, ExplorationWorksCorrectly) {
    std::vector<double> true_probs = {0.2, 0.8};
    EpsilonGreedyAgent agent(true_probs, 1.0, 2);
    agent.run_simulation(100);
    
    // The number of pulls should be roughly equal for both arms
    ASSERT_NEAR(agent.get_bandit(0).get_pull_count(), 50, 15);
    ASSERT_NEAR(agent.get_bandit(1).get_pull_count(), 50, 15);
}

// Test case 1: Ensure initial pulls are one for each arm
TEST(UCBAgentTest, AllArmsPulledOnceInitially) {
    std::vector<double> true_probs = {0.1, 0.2, 0.3};
    UCBAgent agent(true_probs, 2.0);
    
    // Each arm should be pulled once in the first 3 steps
    agent.run_simulation(3);
    
    ASSERT_EQ(agent.get_bandit(0).get_pull_count(), 1);
    ASSERT_EQ(agent.get_bandit(1).get_pull_count(), 1);
    ASSERT_EQ(agent.get_bandit(2).get_pull_count(), 1);
}
