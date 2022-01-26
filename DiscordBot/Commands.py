import discord
from discord.ext import commands
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import requests
from io import BytesIO
class HebrewNet(nn.Module):
    def __init__(self):
        super(HebrewNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32, 256),
            nn.Sigmoid(),
            #nn.Linear(512, 512),
            #nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 4),
            
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Scribe(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self._last_member = None

    @commands.command()
    async def predict(self, ctx):
        """Says hello"""
        model = HebrewNet()
        model.load_state_dict(torch.load('default.model'))
        model.eval()

        response = requests.get(ctx.message.attachments[0].url)
        image = Image.open(BytesIO(response.content)).convert('L')
        image = image.resize((32, 32))
        image = np.array(image)
        classes = ['alef', 'het', 'mem', 'shin']

        x = torch.from_numpy(image).float()
        x = (x.unsqueeze(0).unsqueeze(0))

        prediction = model(x)
        perc = nn.functional.softmax(prediction[0], dim=0)
        i = np.argmax(perc.detach().numpy())
        await ctx.send(f'{classes[i]} ({perc[i]*100:>0.1f}%)')