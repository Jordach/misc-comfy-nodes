inputs = ["a", "b", "c", "d"]

def make_selector(type):
	return {
		"required": {
			"slot": (inputs,),
			"a": (type,)
		},
		"optional": {
			"b": (type,),
			"c": (type,),
			"d": (type,),
		}
	}

def ez_select(slot, a, b=None, c=None, d=None):
	if slot == "a":
		return (a,)
	elif slot == "b":
		if b is None:
			return (a,)
		else:
			return (b,)
	elif slot == "c":
		if c is None:
			return (a,)
		else:
			return (c,)
	elif slot == "d":
		if d is None:
			return (a,)
		else:
			return (d,)

class LatentSelector:
	@classmethod
	def INPUT_TYPES(self):
		return make_selector("LATENT")
	
	RETURN_TYPES = ("LATENT",)
	FUNCTION = "selector"
	CATEGORY = "utils"

	def selector(self, slot, a, b=None, c=None, d=None):
		return ez_select(slot, a, b, c, d)

class CondSelector:
	@classmethod
	def INPUT_TYPES(self):
		return make_selector("CONDITIONING")
	
	RETURN_TYPES = ("CONDITIONING",)
	FUNCTION = "selector"
	CATEGORY = "utils"

	def selector(self, slot, a, b=None, c=None, d=None):
		return ez_select(slot, a, b, c, d)

class StringSelector:
	@classmethod
	def INPUT_TYPES(self):
		return make_selector("STRING")
	
	RETURN_TYPES = ("STRING",)
	FUNCTION = "selector"
	CATEGORY = "utils"

	def selector(self, slot, a, b=None, c=None, d=None):
		return ez_select(slot, a, b, c, d)

NODE_CLASS_MAPPINGS = {
	"JDC_LatentSelector": LatentSelector,
	"JDC_CondSelector": CondSelector,
	"JDC_StringSelector": StringSelector,
}


NODE_DISPLAY_NAME_MAPPINGS = {
	"JDC_LatentSelector": "Latent Selector",
	"JDC_CondSelector": "Conditioning Selector",
	"JDC_StringSelector": "String Selector",
}