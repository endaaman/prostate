
model = Model() # the model should be defined with the same code you used to create the trained model
state_dict = torch.load( "./torch_model_v1.pt")
model.load_state_dict(state_dict)
